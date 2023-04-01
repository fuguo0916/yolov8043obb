# Ultralytics YOLO 🚀, GPL-3.0 license
# Checked by FG 20230310

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List

import numpy as np
import torch

from .ops import ltwh2xywh, ltwh2xyxy, resample_segments, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh
from .ops import xyxyxyxya2xywha, xywha2xyxyxyxya, incomplete_xyxyxy2wh, incomplete_xyxyxy2xy


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_4tuple = _ntuple(4)

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(yolo format)
# `ltwh` means left top and width, height(coco format)
_formats = ['xyxy', 'xywh', 'ltwh', 'poly_theta']

__all__ = ['Bboxes']


class Bboxes:
    """Now only numpy is supported"""

    def __init__(self, bboxes, format='poly_theta') -> None:
        """FG
        Bboxes.bboxes is all with shape (n, 9)
        """
        assert format in _formats
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 9
        self.bboxes = bboxes
        self.format = format
        # self.normalized = normalized

    # def convert(self, format):
    #     assert format in _formats
    #     if self.format == format:
    #         bboxes = self.bboxes
    #     elif self.format == "xyxy":
    #         if format == "xywh":
    #             bboxes = xyxy2xywh(self.bboxes)
    #         else:
    #             bboxes = xyxy2ltwh(self.bboxes)
    #     elif self.format == "xywh":
    #         if format == "xyxy":
    #             bboxes = xywh2xyxy(self.bboxes)
    #         else:
    #             bboxes = xywh2ltwh(self.bboxes)
    #     else:
    #         if format == "xyxy":
    #             bboxes = ltwh2xyxy(self.bboxes)
    #         else:
    #             bboxes = ltwh2xywh(self.bboxes)
    #
    #     return Bboxes(bboxes, format)

    def convert(self, format):
        """FG
        It converts self.bboxes and self.format to format.
        """
        assert format in _formats
        if self.format == format:
            return
        elif self.format == 'poly_theta':
            assert False
        elif self.format == 'xywha':
            assert False
            # assert format == "xyxyxyxya"
            # bboxes = xywha2xyxyxyxya(self.bboxes)
            # assert len(bboxes.shape) == 2 and bboxes.shape[0] == self.bboxes.shape[0] and bboxes.shape[1] == 9
        elif self.format == 'xyxyxyxya':
            assert False
            # assert format == "xywha"
            # bboxes = xyxyxyxya2xywha(self.bboxes)
            # assert len(bboxes.shape) == 2 and bboxes.shape[0] == self.bboxes.shape[0] and bboxes.shape[1] == 5
        elif self.format == 'xyxy':
            bboxes = xyxy2xywh(self.bboxes) if format == 'xywh' else xyxy2ltwh(self.bboxes)
        elif self.format == 'xywh':
            bboxes = xywh2xyxy(self.bboxes) if format == 'xyxy' else xywh2ltwh(self.bboxes)
        else:
            bboxes = ltwh2xyxy(self.bboxes) if format == 'xyxy' else ltwh2xywh(self.bboxes)
        self.bboxes = bboxes
        self.format = format

    def areas(self):
        self.convert('xyxy')
        return (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])

    # def denormalize(self, w, h):
    #     if not self.normalized:
    #         return
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #     self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h
    #     self.normalized = True

    def mul(self, scale):
        """
        Args:
            scale (tuple | List | int): the scale for 9 coords.
        """
        assert self.format == "poly_theta"
        if isinstance(scale, Number):
            scale = (_ntuple(9))(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 9
        for i in range(9):
            self.bboxes[:, i] *= scale[i]

    def add(self, offset):
        """
        Args:
            offset (tuple | List | int): the offset for four coords.
        """
        k = 5 if self.format == "xywha" else 9
        if isinstance(offset, Number):
            offset = (_ntuple(k))(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == k
        for i in range(k):
            self.bboxes[:, i] += offset[i]

    def __len__(self):
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: List['Bboxes'], axis=0) -> 'Bboxes':
        """
        Concatenates a list of Boxes into a single Bboxes

        Arguments:
            boxes_list (list[Bboxes])

        Returns:
            Bboxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index) -> 'Bboxes':
        """
        Args:
            index: int, slice, or a BoolArray

        Returns:
            Bboxes: Create a new :class:`Bboxes` by indexing.
        """
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].view(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f'Indexing on Bboxes with {index} failed to return a matrix!'
        return Bboxes(b)


class Instances:

    def __init__(self, bboxes, segments=None, keypoints=None, bbox_format='xywh', normalized=True) -> None:
        """
        Args:
            bboxes (ndarray): bboxes with shape [N, 4].
            segments (list | ndarray): segments.
            keypoints (ndarray): keypoints with shape [N, 17, 2].
        """
        if segments is None:
            segments = []
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format)
        self.keypoints = keypoints
        self.normalized = normalized

        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            segments = resample_segments(segments)
            # (N, 1000, 2)
            segments = np.stack(segments, axis=0)
        else:
            segments = np.zeros((0, 1000, 2), dtype=np.float32)
        self.segments = segments

    def convert_bbox(self, format):
        self._bboxes.convert(format=format)

    def bbox_areas(self):
        self._bboxes.areas()

    def scale(self, scale_w, scale_h, bbox_only=False):
        """this might be similar with denormalize func but without normalized sign"""
        assert self._bboxes.format == "poly_theta"
        self._bboxes.mul(scale=(scale_w, scale_h) * 4 + (1,))
        if bbox_only:
            return
        self.segments[..., 0] *= scale_w
        self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w, h):
        if not self.normalized:
            return
        assert self._bboxes.format == "poly_theta"
        self._bboxes.mul(scale=(w, h) * 4 + (1,))
        self.segments[..., 0] *= w
        self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w, h):
        if self.normalized:
            return
        assert self._bboxes.format == "poly_theta"
        self._bboxes.mul(scale=(1/w, 1/h) * 4 + (1,))
        self.segments[..., 0] /= w
        self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw, padh):
        # handle rect and mosaic situation
        assert not self.normalized, 'you should add padding with absolute coordinates.'
        assert self._bboxes.format == 'poly_theta'
        self._bboxes.add((padw, padh, padw, padh, padw, padh, padw, padh, 0))
        self.segments[..., 0] += padw
        self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index) -> 'Instances':
        """
        Args:
            index: int, slice, or a BoolArray

        Returns:
            Instances: Create a new :class:`Instances` by indexing.
        """
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index]
        bbox_format = self._bboxes.format
        return Instances(
            bboxes=bboxes,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    # points mapping: 1234 -> 2143
    # angle = - angle
    def flipud(self, h):
        if self._bboxes.format == 'poly_theta':
            y1 = self.bboxes[:, 1].copy()
            y2 = self.bboxes[:, 3].copy()
            y3 = self.bboxes[:, 5].copy()
            y4 = self.bboxes[:, 7].copy()
            self.bboxes[:, 1] = h - y1
            self.bboxes[:, 3] = h - y2
            self.bboxes[:, 5] = h - y3
            self.bboxes[:, 7] = h - y4
            self.bboxes[:, :8] = self.bboxes[:, [2, 3, 0, 1, 6, 7, 4, 5]]
            self.bboxes[:, 8] = self.bboxes[:, 8] * (-1)
        else:
            assert False, f"wrong box format {self._bboxes.format}"
            self.bboxes[:, 1] = h - self.bboxes[:, 1]
            self.bboxes[:, 4] = self.bboxes[:, 4] * (-1)
        self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    # points mapping: 1234 -> 4321
    # angle = PI - angle
    def fliplr(self, w):
        if self._bboxes.format == 'poly_theta':
            x1 = self.bboxes[:, 0].copy()
            x2 = self.bboxes[:, 2].copy()
            x3 = self.bboxes[:, 4].copy()
            x4 = self.bboxes[:, 6].copy()
            self.bboxes[:, 0] = w - x1
            self.bboxes[:, 2] = w - x2
            self.bboxes[:, 4] = w - x3
            self.bboxes[:, 6] = w - x4
            self.bboxes[:, :8] = self.bboxes[:, [6, 7, 4, 5, 2, 3, 0, 1]]
            self.bboxes[:, 8] = -self.bboxes[:, 8]
        else:
            assert False, f"wrong box format {self._bboxes.format}"
            self.bboxes[:, 0] = w - self.bboxes[:, 0]
            self.bboxes[:, 4] = -self.bboxes[:, 4]
        self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w, h):
        assert self._bboxes.format == "poly_theta"
        assert not self.normalized
        assert isinstance(self._bboxes.bboxes, np.ndarray)
        for i in range(self._bboxes.bboxes.shape[0]):
            if (self.bboxes[i][:8] >= 0).all() and (self.bboxes[i][[0, 2, 4, 6]] <= w).all() and (self.bboxes[i][[1, 3, 5, 7]] <= h).all():
                pass
            else:
                def clip_obb(obb):
                    """FG
                    Args:
                        obb: ndarray(9,) pixel-polytheta
                    Return:
                        obb: ndarray(9,) clipped

                    The `obb` arg might be modified during the function call.
                    """
                    x1, y1, x2, y2, x3, y3, x4, y4, angle = obb
                    p1in = x1 >= 0 and x1 <= w and y1 >= 0 and y1 <= h
                    p2in = x2 >= 0 and x2 <= w and y2 >= 0 and y2 <= h
                    p3in = x3 >= 0 and x3 <= w and y3 >= 0 and y3 <= h
                    p4in = x4 >= 0 and x4 <= w and y4 >= 0 and y4 <= h
                    in_num = sum([1 if x else 0 for x in [p1in, p2in, p3in, p4in]])
                    if in_num == 3:
                        pass
                    elif in_num == 2:
                        def translate_line_segment(v1, v2, direction):
                            if direction[0] != 0 and direction[1] != 0:
                                dx, dy = direction
                                x1, y1 = v1
                                x2, y2 = v2
                                translate_scales1 = [(0 - x1) / dx, (h - y1) / dy, (w - x1) / dx, (0 - y1) / dy]
                                translate_scales2 = [(0 - x2) / dx, (h - y2) / dy, (w - x2) / dx, (0 - y2) / dy]
                                min1 = sorted(translate_scales1)[1]
                                min2 = sorted(translate_scales2)[1]
                                min1_index = translate_scales1.index(min1)
                                min2_index = translate_scales2.index(min2)
                                if min1_index == min2_index:
                                    translate_scale = min(min1, min2)
                                    translate_dx = dx * translate_scale
                                    translate_dy = dy * translate_scale
                                    v1 = (v1[0] + translate_dx, v1[1] + translate_dy)
                                    v2 = (v2[0] + translate_dx, v2[1] + translate_dy)
                                else:
                                    cross_list = [False, False, False, False]
                                    cross_list[min1_index] = True
                                    cross_list[min2_index] = True
                                    if cross_list[0] and cross_list[1]:
                                        corner = (0, h)
                                    elif cross_list[1] and cross_list[2]:
                                        corner = (w, h)
                                    elif cross_list[2] and cross_list[3]:
                                        corner = (w, 0)
                                    elif cross_list[3] and cross_list[0]:
                                        corner = (0, 0)
                                    else:
                                        assert False, f"\nInstances.clip: wrong cross_list: {cross_list}\n"
                                    
                                    ## line: y = kx + b
                                    ## perpendicular: y = -1/k *(x - x0) + y0
                                    ## foot x: [k*(y0 - b) + x0] / (1 + k^2) 
                                    k = (y1 - y2) / (x1 - x2)
                                    b = y1 - x1 * k
                                    foot_x = (k * (corner[1] - b) + corner[0]) / (1 + k**2)
                                    scale3 = (corner[0] - foot_x) / dx
                                    if scale3 <= 0:
                                        pass
                                    else:
                                        # assert scale3 <= min1 and scale3 <= min2
                                        translate_scale = scale3
                                        translate_dx = dx * translate_scale
                                        translate_dy = dy * translate_scale
                                        v1 = (v1[0] + translate_dx, v1[1] + translate_dy)
                                        v2 = (v2[0] + translate_dx, v2[1] + translate_dy)
                            else:
                                assert not direction[0] == 0 and direction[1] == 0
                                if direction[0] == 0:
                                    if direction[1] > 0:
                                        v1[1] = 0
                                        v2[1] = 0
                                    else:
                                        v1[1] = h
                                        v2[1] = h
                                else:
                                    if direction[0] > 0:
                                        v1[0] = 0
                                        v2[0] = 0
                                    else:
                                        v1[0] = w
                                        v2[0] = w
                                
                            return v1, v2
                        
                        if not p1in and not p2in: 
                            (x1, y1), (x2, y2) = translate_line_segment((x1, y1), (x2, y2), (x3 - x2, y3 - y2))
                        elif not p2in and not p3in:
                            (x2, y2), (x3, y3) = translate_line_segment((x2, y2), (x3, y3), (x4 - x3, y4 - y3))
                        elif not p3in and not p4in:
                            (x3, y3), (x4, y4) = translate_line_segment((x3, y3), (x4, y4), (x1 - x4, y1 - y4))
                        elif not p4in and not p1in:
                            (x4, y4), (x1, y1) = translate_line_segment((x4, y4), (x1, y1), (x2 - x1, y2 - y1))
                        else:
                            pass
                        obb[:9] = [x1, y1, x2, y2, x3, y3, x4, y4, angle]
                    else:
                        pass
                    return obb
                
                self._bboxes.bboxes[i] = clip_obb(self._bboxes.bboxes[i])
                

    def obb_filter(self, w, h):
        """
        It returns a mask for boxes whose center is in (w,h)
        """
        assert not self.normalized
        xy = incomplete_xyxyxy2xy(self.bboxes[:, :6])
        assert isinstance(xy, np.ndarray)
        a = np.logical_and
        mask = a(a(a((xy[:, 0] >= 0), (xy[:, 0] <= w)), (xy[:, 1] >= 0)), (xy[:, 1] <= h))
        return mask


    def update(self, bboxes, segments=None, keypoints=None):
        new_bboxes = Bboxes(bboxes, format=self._bboxes.format)
        self._bboxes = new_bboxes
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self):
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, instances_list: List['Instances'], axis=0) -> 'Instances':
        """
        Concatenates a list of Boxes into a single Bboxes

        Arguments:
            instances_list (list[Bboxes])
            axis

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis)
        cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis)
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self):
        return self._bboxes.bboxes
