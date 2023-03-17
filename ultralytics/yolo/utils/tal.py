# Ultralytics YOLO 🚀, GPL-3.0 license
# Checked by FG 20230310

import torch
import torch.nn as nn
import torch.nn.functional as F

from .checks import check_version
from .metrics import bbox_iou, obb_iou

TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9, roll_out=False):
    """select the positive anchor center in gt

    Args:
        xy_centers (Tensor): shape(h*w, 4)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)
    Return:
        (Tensor): shape(b, n_boxes, h*w)

    Args:
        xy_centers: Tensor(na, 2) Pixel
        gt_bboxes: Tensor(bs, n_max_gt, 9) Pixel-Poly
    Return:
        Tensor(bs, n_max_gt, na) 1/0 if xy_centers in gt_bboxes
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    if roll_out:
        bbox_deltas = torch.empty((bs, n_boxes, n_anchors), device=gt_bboxes.device)
        for b in range(bs):
            lt, rb = gt_bboxes[b].view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
            bbox_deltas[b] = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]),
                                       dim=2).view(n_boxes, n_anchors, -1).amin(2).gt_(eps)
        return bbox_deltas
    else:
        lt, lb, rb, rt = gt_bboxes[..., :8].view(-1, 1, 8).chunk(chunks=4, dim=2)
        # bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        # return bbox_deltas.amin(3).gt_(eps)
        vec_w = rt - lt
        vec_h = lb - lt
        from_lt = xy_centers[None] - lt
        from_lb = xy_centers[None] - lb
        from_rb = xy_centers[None] - rb
        from_rt = xy_centers[None] - rt
        res = (from_lt * vec_w).sum(dim=-1).gt_(eps) + (from_lt * vec_h).sum(dim=-1).gt_(eps)\
            + (from_lb * vec_w).sum(dim=-1).gt_(eps) + (-from_lb * vec_h).sum(dim=-1).gt_(eps)\
            + (-from_rb * vec_w).sum(dim=-1).gt_(eps) + (-from_rb * vec_h).sum(dim=-1).gt_(eps)\
            + (-from_rt * vec_w).sum(dim=-1).gt_(eps) + (from_rt * vec_h).sum(dim=-1).gt_(eps)
        res = res.ge_(8).view(bs, n_boxes, n_anchors)
        return res



def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)
    Return:
        target_gt_idx: Tensor(bs, na)
        fg_mask: Tensor(bs, na) 1/0 if matches a gt
        mask_pos: Tensor(bs, n_max_gt, na)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)  # (b, n_max_boxes, h*w)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    # find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, roll_out_thr=0):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.roll_out_thr = roll_out_thr

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores: Tensor(bs, na, nc) 0-1
            pd_bboxes: Tensor(bs, na, 9) Pixel-Poly
            anc_points: Tensor(na, 2) Pixel
            gt_labels: Tensor(bs, n_max_gt, 1)
            gt_bboxes: Tensor(bs, n_max_gt, 9) Pixel-Poly
            mask_gt: Tensor(bs, n_max_gt, 1)
        Returns:
            target_labels: Tensor(bs, na) cls_idx
            target_bboxes: Tensor(bs, na, 9) Pixel-Poly
            target_scores: Tensor(bs, na, nc) overlap roughly
            fg_mask: Tensor(bs, na) T/F if matched
            target_gt_idx: Tensor(bs, na) matched gt idx
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        self.roll_out = self.n_max_boxes > self.roll_out_thr if self.roll_out_thr else False

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.bg_idx).to(device), torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device), torch.zeros_like(pd_scores[..., 0]).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))
            # target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points,
                                                             mask_gt)
        """FG
            mask_pos: Tensor(bs, n_max_gt, na) 1/0 positive mask. Requirements: ap in gt; pred in gt's topk by align_metric
            align_metric: Tensor(bs, n_max_gt, na). From overlaps and bbox_scores.
            overlaps: Tensor(bs, n_max_gt, na) 0-1
        """

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        """FG
            当多个GT匹配到一个Anchor Point的时候，此时选择与此Anchor Point对应的Overlap最大的GT
            target_gt_idx: Tensor(bs, na)
            fg_mask: Tensor(bs, na) 1/0 if matches a gt
            mask_pos: Tensor(bs, n_max_gt, na)
        """

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        """FG
            target_bboxes: (bs, na, 9) Pixel-Poly
            target_labels: (bs, na, 1) cls_idx
            target_scores: (bs, na, nc) 1/0
        """

        # normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  ## (b, n_max_gt, 1)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  ## (b, n_max_gt, 1)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)  ## (b, na, 1)
        target_scores = target_scores * norm_align_metric  ## target_scores *= overlaps roughly

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """FG
        Args:
            pd_scores: Tensor(bs, na, 1) 0-1
            pd_bboxes: Tensor(bs, na, 9) Pixel-Poly
            gt_labels: Tensor(bs, n_max_gt, 1) cls_idx
            gt_bboxes: Tensor(bs, n_max_gt, 9) Pixel-Poly
            anc_points: Tensor(na, 2) Pixel
            mask_gt: Tensor(bs, n_max_gt, 1) 1./0.
        Return:
            mask_pos: Tensor(bs, n_max_gt, na) 1/0 positive mask. Requirements: ap in gt; pred in gt's topk
            align_metric: Tensor(bs, n_max_gt, na). From overlaps and bbox_scores.
            overlaps: Tensor(bs, n_max_gt, na) 0-1
        """
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        ## align_metric, overlaps: both Tensor(bs,n_max_gt,na)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes, roll_out=self.roll_out)
        ## mask_in_gts: Tensor(bs, n_max_gt, na) 1/0 if anchor_points in gt_bboxes
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts,
                                                topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())
        ## mask_topk: Tensor(bs, n_max_gt, na) 1/0 if pred_box is in gt's topk
        # merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        if self.roll_out:
            assert False
            align_metric = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            overlaps = torch.empty((self.bs, self.n_max_boxes, pd_scores.shape[1]), device=pd_scores.device)
            ind_0 = torch.empty(self.n_max_boxes, dtype=torch.long)
            for b in range(self.bs):
                ind_0[:], ind_2 = b, gt_labels[b].squeeze(-1).long()
                # get the scores of each grid for each gt cls
                bbox_scores = pd_scores[ind_0, :, ind_2]  # b, max_num_obj, h*w
                overlaps[b] = bbox_iou(gt_bboxes[b].unsqueeze(1), pd_bboxes[b].unsqueeze(0), xywh=False,
                                       CIoU=True).squeeze(2).clamp(0)
                align_metric[b] = bbox_scores.pow(self.alpha) * overlaps[b].pow(self.beta)
        else:
            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
            ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)  # b, max_num_obj
            ind[1] = gt_labels.long().squeeze(-1)  # b, max_num_obj
            # get the scores of each grid for each gt cls
            bbox_scores = pd_scores[ind[0], :, ind[1]]  # b, max_num_obj, h*w

            overlaps = obb_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1)).squeeze(3).clamp(0)  # bbox_iou
            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps  ## both Tensor(bs, n_max_gt, na)

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """
        Args:
            metrics: Tensor(bs, n_max_gt, na). Value from (align_metric * mask_in_gts)
            topk_mask: Tensor(bs, n_max_gt, k). Value from broadcast of mask_gt (bs, n_max_gt)
        Return:
            is_in_topk: Tensor(bs, n_max_gt, na) 1/0
        """

        num_anchors = metrics.shape[-1]  # h*w
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)  ## (bs, n_max_gt, k)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        # (b, max_num_obj, topk)
        topk_idxs[~topk_mask] = 0
        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        if self.roll_out:
            is_in_topk = torch.empty(metrics.shape, dtype=torch.long, device=metrics.device)
            for b in range(len(topk_idxs)):
                is_in_topk[b] = F.one_hot(topk_idxs[b], num_anchors).sum(-2)
        else:
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(-2)
        # filter invalid bboxes
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (b, max_num_obj, 1)
            gt_bboxes: (b, max_num_obj, 9) Pixel-Poly
            target_gt_idx: (b, h*w)
            fg_mask: (b, h*w)
        Return:
            target_bboxes: (bs, na, 9) Pixel-Poly
            target_labels: (bs, na, 1) cls_idx
            target_scores: (bs, na, nc) 1/0
        """

        # assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # assigned target boxes, (b, max_num_obj, 4) -> (b, h*w)
        target_bboxes = gt_bboxes.view(-1, 9)[target_gt_idx]

        # assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (b, h*w, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, angle, anchor_points, dim=-1):
    """FG
    Args:
        distance: e.g. Tensor(bs, na, 4) Grid relative
        angle: e.g. Tensor(bs, na, 1)
        anchor_points: e.g. Tensor(na, 2) Grid relative
        dim: the dim index of 4,1,2
    Return:
        Grid relative box
    """""
    # assert dim == -1
    left, top, right, bottom = distance.chunk(4, dim)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    x, y = anchor_points.chunk(2, dim)
    lsin, tsin, rsin, bsin = (distance * sin_a).chunk(4, dim)
    lcos, tcos, rcos, bcos = (distance * cos_a).chunk(4, dim)
    x1 = x - lcos + tsin
    y1 = y - lsin - tcos
    x2 = x - lcos - bsin
    y2 = y - lsin + bcos
    x3 = x + rcos - bsin
    y3 = y + rsin + bcos
    x4 = x + rcos + tsin
    y4 = y + rsin - tcos
    return torch.cat((x1, y1, x2, y2, x3, y3, x4, y4, angle), dim)


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    """FG
    Args:
        anchor_points: Tensor(na, 2) Grid relative
        bbox: (bs, na, 9) Grid-Poly
        reg_max: int
    Return:
        dist: (bs, na, 4) Grid ltrb
    """
    x = anchor_points[..., 0]
    y = anchor_points[..., 1]
    x1, y1 = bbox[..., 0], bbox[..., 1]
    x3, y3 = bbox[..., 4], bbox[..., 5]
    angle = bbox[..., -1]
    cos_a, sin_a = torch.cos(angle), torch.sin(angle)
    l = (x-x1) * cos_a + (y-y1) * sin_a
    t = -(x-x1) * sin_a + (y-y1) * cos_a
    b = (x-x3) * sin_a - (y-y3) * cos_a
    r = -(x-x3) * cos_a - (y-y3) * sin_a
    dist = torch.stack((l, t, r, b), dim=-1)
    dist = dist.clamp(0, reg_max - 0.01)
    return dist
    # x1y1, x2y2 = bbox.chunk(2, -1)
    # return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)
