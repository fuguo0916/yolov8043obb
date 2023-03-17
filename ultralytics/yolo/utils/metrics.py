# Ultralytics YOLO 🚀, GPL-3.0 license
# Checked by FG 20230310
"""
Model validation metrics
"""
import math
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ultralytics.yolo.utils import LOGGER, TryExcept
from ultralytics.yolo.utils.piou_loss.pixel_weights import PIoU, Pious, template_pixels, template_w_pixels


# boxes
def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def bbox_ioa(box1, box2, eps=1e-7):
    """Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(nx4)
    box2:       np.array of shape(mx4)
    returns:    np.array of shape(nxm)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    assert len(box1.shape) == 2 and len(box2.shape) == 2
    assert box1.shape[-1] == 4 and box2.shape[-1] == 4

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    iou = inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
    assert len(iou.shape) == 2 and iou.shape[0] == box1.shape[0] and iou.shape[1] == box2.shape[0]
    return iou


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU


def get_enclosing_hbb(box):
    """FG
    Args:
        box: obb(s) in unknown format
    Return:
        hbb(s) in Poly format
    """
    assert box.shape[-1] == 9
    xmin = torch.min(box[..., [0, 2, 4, 6]], dim=-1, keepdim=True)[0]
    xmax = torch.max(box[..., [0, 2, 4, 6]], dim=-1, keepdim=True)[0]
    ymin = torch.min(box[..., [1, 3, 5, 7]], dim=-1, keepdim=True)[0]
    ymax = torch.max(box[..., [1, 3, 5, 7]], dim=-1, keepdim=True)[0]
    hbb = torch.cat([xmin, ymin, xmax, ymax], dim=-1)
    assert hbb.shape[:-1] == box.shape[:-1] and hbb.shape[-1] == 4
    return hbb


# https://github.com/open-mmlab/mmrotate
def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


# https://github.com/open-mmlab/mmrotate
def kfiou(pred, target, pred_decode=None, targets_decode=None, beta=1.0 / 9.0, eps=1e-6):
    """Kalman filter IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.

    Returns:
        kfiou (torch.Tensor)
    """
    xy_p = pred[:, :2]
    xy_t = target[:, :2]
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred_decode)
    _, Sigma_t = xy_wh_r_2_xy_sigma(targets_decode)

    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                        diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)

    # if fun == 'ln':
    #     kf_loss = -torch.log(KFIoU + eps)
    # elif fun == 'exp':
    #     kf_loss = torch.exp(1 - KFIoU) - 1
    # else:
    #     kf_loss = 1 - KFIoU

    # loss = (xy_loss + kf_loss).clamp(0)

    return KFIoU
    


def obb_iou(box1, box2, eps=1e-7, choice="miou"):
    """FG
    Args:
        box1: shape (..., 9)
        box2: shape (..., 9)
    Return:
        obb_iou with shape (..., 1)
    """
    assert box1.shape[-1] == box2.shape[-1]
    assert box1.shape[-1] == 9
    if choice == "miou":
        angle1 = box1[..., -1:]
        angle2 = box2[..., -1:]
        angle_difference = angle1 - angle2
        angle_similarity = torch.abs(torch.cos(angle_difference))
        box1 = get_enclosing_hbb(box1)
        box2 = get_enclosing_hbb(box2)
        hbb_iou = bbox_iou(box1, box2, xywh=False, eps=eps)
        iou = hbb_iou * angle_similarity
    elif choice == "kfiou":
        def xyxyxyxya2xywha(x):
            """FG
            Args:
                x: boxes, the last dim is 9
            Return:
                y with almost same shape
            """
            if isinstance(x, torch.Tensor):
                assert x.shape[-1] == 9
                y = x.clone()
                y.resize_(list(x.shape[:-1]) + [5,])
                y[..., 0:2] = (x[..., 0:2] + x[..., 4:6]) / 2.0  # xy
                y[..., 2] = torch.sqrt(torch.pow(x[..., 2] - x[..., 4], 2) + torch.pow(x[..., 3] - x[..., 5], 2))  # w: distance between x2y2 and x3y3
                y[..., 3] = torch.sqrt(torch.pow(x[..., 0] - x[..., 2], 2) + torch.pow(x[..., 1] - x[..., 3], 2))  # h: distance between x1y1 and x2y2
                y[..., 4] = x[..., 8]
            else:
                assert len(x.shape) == 2 and x.shape[-1] == 9
                y = np.copy(x)
                y.resize((x.shape[0], 5), refcheck=False)
                y[:, 0:2] = (x[:, 0:2] + x[:, 4:6]) / 2.0  # xy
                y[:, 2] = np.sqrt(np.power(x[:, 2] - x[:, 4], 2) + np.power(x[:, 3] - x[:, 5], 2))  # w
                y[:, 3] = np.sqrt(np.power(x[:, 0] - x[:, 2], 2) + np.power(x[:, 1] - x[:, 3], 2))  # h
                y[:, 4] = x[:, 8]
            assert y.shape[:-1] == x.shape[:-1] and y.shape[-1] == 5
            return y

        box1 = xyxyxyxya2xywha(box1)
        box2 = xyxyxyxya2xywha(box2)
        print(f"box1: {box1.shape}\nbox2: {box2.shape}")
        iou = kfiou(box1, box2, box1, box2)
        assert False, f"\nbox1: {box1.shape}\nbox2: {box2.shape}\nkfiou: {iou.shape}"
        
    assert iou.shape[-1] == 1
    return iou

        



def obb_iou_for_metrics(box1, box2, choice="rotated_iou", eps=1e-7):
    """FG
    Args:
        box1: obb with shape (N, 5/9)
        box2: obb with shape (M, 5/9)
    Return:
        iou between box1 and box2 with shape (N, M)
    """
    def xyxyxyxya2xywha(x):
        """FG
        Args:
            x: boxes, the last dim is 9
        Return:
            y with almost same shape
        """
        if isinstance(x, torch.Tensor):
            assert x.shape[-1] == 9
            y = x.clone()
            y.resize_(list(x.shape[:-1]) + [5,])
            y[..., 0:2] = (x[..., 0:2] + x[..., 4:6]) / 2.0  # xy
            y[..., 2] = torch.sqrt(torch.pow(x[..., 2] - x[..., 4], 2) + torch.pow(x[..., 3] - x[..., 5], 2))  # w: distance between x2y2 and x3y3
            y[..., 3] = torch.sqrt(torch.pow(x[..., 0] - x[..., 2], 2) + torch.pow(x[..., 1] - x[..., 3], 2))  # h: distance between x1y1 and x2y2
            y[..., 4] = x[..., 8]
        else:
            assert len(x.shape) == 2 and x.shape[-1] == 9
            y = np.copy(x)
            y.resize((x.shape[0], 5), refcheck=False)
            y[:, 0:2] = (x[:, 0:2] + x[:, 4:6]) / 2.0  # xy
            y[:, 2] = np.sqrt(np.power(x[:, 2] - x[:, 4], 2) + np.power(x[:, 3] - x[:, 5], 2))  # w
            y[:, 3] = np.sqrt(np.power(x[:, 0] - x[:, 2], 2) + np.power(x[:, 1] - x[:, 3], 2))  # h
            y[:, 4] = x[:, 8]
        assert y.shape[:-1] == x.shape[:-1] and y.shape[-1] == 5
        return y

    if choice == "miou":
        return obb_iou_plain(box1, box2, eps=1e-7)
    elif choice in ["piou", "piou_python", "piou_cuda"]:
        assert torch.max(box1) > 2.0
        assert isinstance(box1, torch.Tensor) and isinstance(box2, torch.Tensor)
        
        width = torch.max(torch.cat((box1[:,[0, 2, 4, 6]], box2[:,[0, 2, 4, 6]]), dim=0)).int()
        height = torch.max(torch.cat((box1[:,[1, 3, 5, 7]], box2[:,[1, 3, 5, 7]]), dim=0)).int()
        grid_xy = template_pixels(height=height, width=width).to(box1.device)
        grid_x = template_w_pixels(width=width).to(box1.device)

        N = box1.shape[0]
        M = box2.shape[0]
        
        box1 = xyxyxyxya2xywha(box1)
        box2 = xyxyxyxya2xywha(box2)
        if choice == "piou_python":
            pious = box1.clone().resize_((N, M))
            for i in range(M):
                _, piou = PIoU(box1, box2[i].unsqueeze(0).repeat(N, 1).contiguous(), grid_xy, k=10)
                # print(f"piou: {piou.data}")
                pious[:, i] = piou
            return pious
        else:
            pious = box1.clone().resize_((N, M))
            PiousF = Pious(10)
            for i in range(M):
                piou = PiousF(box1, box2[i].unsqueeze(0).repeat(N, 1).contiguous(), grid_x)
                # print(f"piou: {piou.data}")
                pious[:, i] = piou
            return pious
    elif choice in ["rotated_iou"]:
        from mmcv.ops import diff_iou_rotated_2d

        assert torch.max(box1) > 2.0
        assert isinstance(box1, torch.Tensor) and isinstance(box2, torch.Tensor)
        
        N = box1.shape[0]
        M = box2.shape[0]
        
        box1 = xyxyxyxya2xywha(box1)
        box2 = xyxyxyxya2xywha(box2)
        pious = box1.clone().resize_((N, M))
        for i in range(M):
            piou = diff_iou_rotated_2d(box1.unsqueeze(0), box2[i].unsqueeze(0).repeat(N, 1).contiguous().unsqueeze(0)).view(-1)
            # print(f"piou: {piou.data}")
            pious[:, i] = piou
        return pious

    else:
        assert False


def obb_iou_plain(box1, box2, eps=1e-7):
    """FG
    Args:
        box1: obb with shape (N, 5/9)
        box2: obb with shape (M, 5/9)
    Return:
        plain iou between box1 and box2 with shape (N, M)
    """
    assert len(box1.shape) == 2 and len(box2.shape) == 2
    assert box1.shape[-1] == box2.shape[-1]
    assert box1.shape[-1] == 9
    angle1 = box1[..., -1]
    angle2 = box2[..., -1]
    angle_difference = angle1.unsqueeze(1) - angle2.unsqueeze(0)
    angle_similarity = torch.abs(torch.cos(angle_difference))
    box1 = get_enclosing_hbb(box1)
    box2 = get_enclosing_hbb(box2)
    hbb_iou = box_iou(box1, box2)
    iou = hbb_iou * angle_similarity
    assert len(iou.shape) == 2 and iou.shape[0] == box1.shape[0] and iou.shape[1] == box2.shape[0]
    return iou


def mask_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [M, n] m2 means number of gt objects
    Note: n means image_w x image_h
    return: masks iou, [N, M]
    """
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def masks_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [N, n] m2 means number of gt objects
    Note: n means image_w x image_h
    return: masks iou, (N, )
    """
    intersection = (mask1 * mask2).sum(1).clamp(0)  # (N, )
    union = (mask1.sum(1) + mask2.sum(1))[None] - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def obb_nms(obbs, iou_threshold):
    hbbs = get_enclosing_hbb(obbs)
    n = obbs.shape[0]
    # if n > 100:
    #     iou_threshold /= 10
    valid_list = [1 for _ in range(n)]
    for i in range(n):
        if valid_list[i] == 1:
            for j in range(i + 1, n):
                if valid_list[j] == 1:
                    hbb_iou = bbox_iou(hbbs[i], hbbs[j], xywh=False)
                    angle_similarity = torch.abs(1 - (torch.abs(obbs[i][8:] - obbs[j][8:]) / 1.5707))
                    if hbb_iou * angle_similarity < iou_threshold:
                        valid_list[j] = 0
    valid_list = torch.tensor(valid_list, device=obbs.device)
    return valid_list


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


# losses
class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in Pixel-PolyTheta format.
        Arguments:
            detections (Array[N, 11]), Pixel-PolyTheta box, conf, class
            labels (Array[M, 10]), class, Pixel-PolyTheta box
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 9] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 10].int()
        iou = obb_iou_for_metrics(labels[:, 1:], detections[:, :9])  # (num_gt, num_pred)

        x = torch.where(iou > self.iou_thres)  # (n_pass, 2)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # n_pass * (gt_idx, pred_idx, iou)
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)  # m0: (n_pass,) gt_idx; m1: (n_pass,) pred_idx
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    @TryExcept('WARNING ⚠️ ConfusionMatrix plot failure')
    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)

    def print(self):
        for i in range(self.nc + 1):
            LOGGER.info(' '.join(map(str, self.matrix[i])))


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=Path(), names=(), eps=1e-16, prefix=''):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10). ndarray(t_npr, niou) T/F
        conf:  Objectness value from 0-1 (nparray). ndarray(t_npr,)
        pred_cls:  Predicted object classes (nparray). ndarray(t_npr,)
        target_cls:  True object classes (nparray). ndarray(t_nl,)
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
        tp: ndarray(nc,) absolute tp numbers
        fp: ndarray(nc,) absolute fp numbers
        p: ndarray(nc,) precision at first iou level at sumit of f1
        r: ndarray(nc,) recall at first iou level at sumit of f1
        f1: ndarray(nc,) the max f1 score at first iou level
        ap: ndarray(nc, niou)
        unique_classes: ndarray(nc,) int
    """

    # Sort by objectness
    i = np.argsort(-conf)
    #
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)  ## ndarray(nc,) clses, ndarray(nc,) counts
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting ## px: ndarray(1000,)
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))  ## ap: ndarray(nc, niou), p: ndarray(nc, 1000), r: ndarray(nc, 1000)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)  ## ndarray(npr, niou)
        tpc = tp[i].cumsum(0)  ## ndarray(npr, niou)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve ## ndarray(npr, niou)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases ## recall points at 1st iou level

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve ## ndarray(npr, niou)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score ## precision points at 1st iou level

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)  ## ndarray(nc, 1000)
    names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, save_dir / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, save_dir / f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, save_dir / f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, save_dir / f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]  ## p,r,f1 is ndarray(nc,) at the sumit of f1 mean
    tp = (r * nt).round()  # true positives ## ndarray(nc,)
    fp = (tp / (p + eps) - tp).round()  # false positives ## ndarray(nc,)
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


class Metric:

    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """AP@0.5 of all classes.
        Return:
            (nc, ) or [].
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """AP@0.5:0.95
        Return:
            (nc, ) or [].
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """mean precision of all classes.
        Return:
            float.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """mean recall of all classes.
        Return:
            float.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """Mean AP@0.5 of all classes.
        Return:
            float.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """Mean AP@0.75 of all classes.
        Return:
            float.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """Mean AP@0.5:0.95 of all classes.
        Return:
            float.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map"""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """class-aware result, return p[i], r[i], ap50[i], ap[i]"""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """mAP of each class"""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        # Model fitness as a weighted combination of metrics
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Args:
            results: tuple(p, r, ap, f1, ap_class)
        """
        self.p, self.r, self.f1, self.all_ap, self.ap_class_index = results


class DetMetrics:

    def __init__(self, save_dir=Path('.'), plot=False, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.names = names
        self.box = Metric()
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

    def process(self, tp, conf, pred_cls, target_cls):
        results = ap_per_class(tp, conf, pred_cls, target_cls, plot=self.plot, save_dir=self.save_dir,
                               names=self.names)[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        return ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

    def mean_results(self):
        return self.box.mean_results()

    def class_result(self, i):
        return self.box.class_result(i)

    @property
    def maps(self):
        return self.box.maps

    @property
    def fitness(self):
        return self.box.fitness()

    @property
    def ap_class_index(self):
        return self.box.ap_class_index

    @property
    def results_dict(self):
        return dict(zip(self.keys + ['fitness'], self.mean_results() + [self.fitness]))


class SegmentMetrics:

    def __init__(self, save_dir=Path('.'), plot=False, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.names = names
        self.box = Metric()
        self.seg = Metric()
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

    def process(self, tp_m, tp_b, conf, pred_cls, target_cls):
        results_mask = ap_per_class(tp_m,
                                    conf,
                                    pred_cls,
                                    target_cls,
                                    plot=self.plot,
                                    save_dir=self.save_dir,
                                    names=self.names,
                                    prefix='Mask')[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        results_box = ap_per_class(tp_b,
                                   conf,
                                   pred_cls,
                                   target_cls,
                                   plot=self.plot,
                                   save_dir=self.save_dir,
                                   names=self.names,
                                   prefix='Box')[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        return [
            'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
            'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']

    def mean_results(self):
        return self.box.mean_results() + self.seg.mean_results()

    def class_result(self, i):
        return self.box.class_result(i) + self.seg.class_result(i)

    @property
    def maps(self):
        return self.box.maps + self.seg.maps

    @property
    def fitness(self):
        return self.seg.fitness() + self.box.fitness()

    @property
    def ap_class_index(self):
        # boxes and masks have the same ap_class_index
        return self.box.ap_class_index

    @property
    def results_dict(self):
        return dict(zip(self.keys + ['fitness'], self.mean_results() + [self.fitness]))


class ClassifyMetrics:

    def __init__(self) -> None:
        self.top1 = 0
        self.top5 = 0
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

    def process(self, targets, pred):
        # target classes and predicted classes
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self):
        return self.top5

    @property
    def results_dict(self):
        return dict(zip(self.keys + ['fitness'], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self):
        return ['metrics/accuracy_top1', 'metrics/accuracy_top5']
