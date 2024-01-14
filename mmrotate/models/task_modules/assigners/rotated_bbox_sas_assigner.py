# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmcv.ops import convex_iou, points_in_polygons
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.base_assigner import BaseAssigner
from mmengine.structures import InstanceData

from mmrotate.registry import TASK_UTILS
from mmrotate.structures.bbox import qbox2hbox,rbox2qbox
from mmdet.structures.bbox import get_box_tensor


def convex_overlaps(gt_rbboxes, points):
    """Compute overlaps between polygons and points.

    Args:
        gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 8).
        points (torch.Tensor): Points to be assigned, shape(n, 18).

    Returns:
        overlaps (torch.Tensor): Overlaps between k gt_bboxes and n bboxes,
            shape(k, n).
    """
    if gt_rbboxes.shape[0] == 0:
        return gt_rbboxes.new_zeros((0, points.shape[0]))
    overlaps = convex_iou(points, gt_rbboxes)
    return overlaps


def AspectRatio(gt_rbboxes):
    """Compute the aspect ratio of all gts.

    Args:
        gt_rbboxes (torch.Tensor): Groundtruth polygons, shape (k, 5).

    Returns:
        ratios (torch.Tensor): The aspect ratio of gt_rbboxes, shape (k, 1).
    """
    edge1 = gt_rbboxes[..., 2]
    edge2 = gt_rbboxes[..., 3]
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    ratios = (width / height)
    return ratios


@TASK_UTILS.register_module()
class rbox_SASAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox. Each proposals will be assigned with `-1` or a positive integer indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        topk (int): number of priors selected in each level
    """

    def __init__(self, topk,iou_calculator):
        self.topk = topk
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(
            self,
            pred_instances: InstanceData,
            num_level_priors: List[int],
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        """Assign gt to bboxes.
        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        计算所有边界框与GT的IoU
        2. compute center distance between all bbox and gt
        计算bbox与GT中心点的距离
        3. on each pyramid level, for each gt, select k bbox whose center are closest to the gt center, so we total select k*l bbox as candidates for each gt
        对于每层的特征金字塔，对于每个GT选择K个距离GT中心点最近的bbox，所以对每个GT总共有K*l个bbox(l:特征金字塔)
        4. get corresponding iou for the these candidates, and compute the mean and std, set mean + std as the iou threshold
        对于候选框，计算均值和方差（标准差？），将mean+std设置为IoU阈值
        5. select these candidates whose iou are greater than or equal to the threshold as positive
        选择候选框中高于/等于IoU阈值的bbox，作为正样本
        6. limit the positive sample's center in gt
        限制正样本的中心点
        参考这里的mmdet3.0的流程（MaxIoU-Assigner）和变量名，已进行修改。
        Args:
            pred_instances (:obj:`InstaceData`): Instances of model predictions. It includes ``priors``, and the priors can be anchors, points, or bboxes predicted by the previous stage, has shape(n, ?).
            num_level_priors (List): Number of priors in each level
            gt_instances (:obj:`InstaceData`): Ground truth of instance annotations. It usually includes ``bboxes`` and ``labels`` attributes.
            gt_instances_ignore (:obj:`InstaceData`, optional): Instances to be ignored during training. It includes ``bboxes`` attribute data that is ignored during training and testing. Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000

        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels

        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None
        # 真实框和候选框的数量
        num_gt, num_priors = gt_bboxes.size(0), priors.size(0)
        # 如果filter bbox了的话不会报错。
        # if priors.shape[0] == 0 or gt_bboxes.shape[0] == 0:
        #     raise ValueError('No gt or bboxes')
        # 1.计算真实框与锚框之间的重叠（IoU）
        overlaps = self.iou_calculator(gt_bboxes, priors).t()
        # overlaps = convex_overlaps(gt_bboxes, priors)
        # 把所有bbox设置为负样本！！！
        assigned_gt_inds = overlaps.new_full((num_priors, ),-1,dtype=torch.long)
        if num_gt == 0 or num_priors == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_priors, ))
            if num_gt == 0:
                # No GT, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_priors,),-1,dtype=torch.long)
            return AssignResult(num_gts=num_gt, gt_inds=assigned_gt_inds, max_overlaps=max_overlaps, labels=assigned_labels)
        # 2. 计算所有bbox和gt之间的中心距离
        # compute center distance between all bbox and gt
        gt_cx = get_box_tensor(gt_bboxes)[:, 0]
        gt_cy = get_box_tensor(gt_bboxes)[:, 1]
        gt_points = torch.stack((gt_cx, gt_cy), dim=1).float()
        # the center of poly
        # gt_bboxes_hbb = qbox2hbox(gt_bboxes)
        #
        # gt_cx = (gt_bboxes_hbb[:, 0] + gt_bboxes_hbb[:, 2]) / 2.0
        # gt_cy = (gt_bboxes_hbb[:, 1] + gt_bboxes_hbb[:, 3]) / 2.0
        # gt_points = torch.stack((gt_cx, gt_cy), dim=1)
        priors_cx = get_box_tensor(priors)[:, 0]
        priors_cy = get_box_tensor(priors)[:, 1]
        priors_points = torch.stack((priors_cx, priors_cy), dim=1)
        # priors = priors.reshape(-1, 9, 2)
        # pts_x = priors[:, :, 0::2]
        # pts_y = priors[:, :, 1::2]
        #
        # pts_x_mean = pts_x.mean(dim=1).squeeze()
        # pts_y_mean = pts_y.mean(dim=1).squeeze()
        #
        # bboxes_points = torch.stack((pts_x_mean, pts_y_mean), dim=1)

        # distances = (bboxes_points[:, None, :] -
        #              gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        # 2.距离公式--计算所有bbox和gt的中心距离
        distances = (priors_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        # Selecting candidates based on the center distance
        # 根据中心距离选择候选框
        # 3.在每个特征金字塔层上，对于每个gt，选择k个中心最接近目标中心的bbox，所以我们总共选择k * l(层数)个bbox作为每个目标的候选框
        candidate_idxs = []
        start_idx = 0
        # 遍历每层的候选框，level表示层，bboxes_per_level:每层的候选框数量
        for level, bboxes_per_level in enumerate(num_level_priors):
            # on each pyramid level, for each gt, select k bbox whose center are closest to the gt center
            # 在每个在金字塔层，对于每个真实框，选择k个锚框中心最接近真实框中心的候选锚框
            end_idx = start_idx + bboxes_per_level
            # 每一层所选的所有锚框的距离（锚框和真实框)
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)
        # 4. get corresponding iou for the these candidates, and compute the mean and std, set mean + std as the iou threshold
        # 为这些候选框获取相应的iou，并计算它们的均值和方差，设置mean+std为iou阈值
        gt_bboxes_ratios = AspectRatio(get_box_tensor(gt_bboxes))
        # 计算每一列的平均值
        gt_bboxes_ratios_per_gt = gt_bboxes_ratios.mean(0)
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        # 候选框的重叠均值
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        # 候选框的重叠方差
        overlaps_std_per_gt = candidate_overlaps.std(0)
        # 候选框的iou阈值
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        # iou阈值权重
        # new assign
        iou_thr_weight = torch.exp((-1 / 4) * gt_bboxes_ratios)
        overlaps_thr_per_gt = overlaps_thr_per_gt * iou_thr_weight
        # clamp neg min threshold--限制负样本最小阈值--未使用
        # overlaps_thr_per_gt = overlaps_thr_per_gt.clamp_min(0.3)
        # 5.选择这些候选框iou大于等于iou阈值的候选框为正样本
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        # 6. limit the positive sample's center in gt
        # 限制正样本的中心在gt
        # from mmdet.ops.point_justify import pointsJf
        # inside_flag = points_in_polygons(bboxes_points, gt_bboxes)
        # gt_bboxes
        inside_flag = points_in_polygons(priors_points, rbox2qbox(get_box_tensor(gt_bboxes)))
        is_in_gts = inside_flag[candidate_idxs,
                                torch.arange(num_gt)].to(is_pos.dtype)
        # 正样本条件：既是正样本并且中心还在gt内
        is_pos = is_pos & is_in_gts
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_priors
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        # 如果一个anchor box被分配给多个目标，则选择iou最高的那个
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]

        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_priors, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
