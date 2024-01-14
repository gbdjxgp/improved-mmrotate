# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import select_single_mlvl
from mmdet.utils import InstanceList, OptInstanceList
from mmengine.config import ConfigDict
from torch import Tensor

from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from ..utils import ORConv2d, RotationInvariantPooling
from .rotated_retina_head import RotatedRetinaHead

from mmdet.structures.bbox import BaseBoxes, cat_boxes, get_box_tensor
from mmdet.models.utils import images_to_levels, multi_apply, unmap
from mmengine.structures import InstanceData
from mmdet.models.task_modules.prior_generators import (AnchorGenerator,anchor_inside_flags)
from .s2a_head import (S2AHead,S2ARefineHead)
from .angle_branch_s2a_head import (AngleBranchS2AHead,AngleBranchS2ARefineHead)
# from .sam_s2a_head import (SAMS2AHead,SAMS2ARefineHead)

@MODELS.register_module()
class AngleBranchSAMS2AHead(AngleBranchS2AHead):
    def get_targets(self,
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True,
                    return_sampling_results: bool = False) -> tuple:
        # 图片数量
        num_imgs = len(batch_img_metas)
        # 确保每张图片都包含一个锚框列表
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        # 计算每个图像的目标
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # anchor number of multi levels-多层（不同大小和比率）锚框数量
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # 1.拼接单图多尺度anchor到一个tensor
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
        # 候选（建议的）的锚框列表 = 不同级别的锚框 * 图片数量
        # NEW
        num_level_proposals_list = [num_level_anchors] * num_imgs
        # compute targets for each image 计算每个图像的目标
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            # NEW
            num_level_proposals_list,
            concat_valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        # :8CHANGED
        rest_results = list(results[7:])
        avg_factor = sum(
            [results.avg_factor for results in sampling_results_list])
        self._raw_positive_infos.update(sampling_results=sampling_results_list)
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,num_level_anchors)
        # NEW CHANGED
        # sam_weights_list = images_to_levels(all_sam_weights, num_level_anchors)
        # res = (labels_list, label_weights_list, bbox_targets_list,
        #        bbox_weights_list, avg_factor,sam_weights_list)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, avg_factor)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)
    def _get_targets_single(self,
                            flat_anchors: Union[Tensor, BaseBoxes],
                            num_level_proposals,#NEW
                            valid_flags: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor or :obj:`BaseBoxes`): Multi-level anchors
                of the image, which are concatenated into a single tensor
                or box type of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors, ).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.  Defaults to True.

        Returns:
            tuple:

                - labels (Tensor): Labels of each level.
                - label_weights (Tensor): Label weights of each level.
                - bbox_targets (Tensor): BBox targets of each level.
                - bbox_weights (Tensor): BBox weights of each level.
                - pos_inds (Tensor): positive samples indexes.
                - neg_inds (Tensor): negative samples indexes.
                - sampling_result (:obj:`SamplingResult`): Sampling results.
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors
        # _get_target_single主要涉及的是assign gt and sample anchors,其中，只对有效anchor去计算target
        anchors = flat_anchors[inside_flags]
        def sam_get_num_level_anchors_inside(num_level_anchors, inside_flags):
            split_inside_flags = torch.split(inside_flags, num_level_anchors)
            num_level_anchors_inside = [
                int(flags.sum()) for flags in split_inside_flags
            ]
            return num_level_anchors_inside

        num_level_anchors_inside =sam_get_num_level_anchors_inside(num_level_proposals, valid_flags)




        pred_instances = InstanceData(priors=anchors)
        # 下面的注释是旧代码复制过来的，我不知道新代码的数据表示格式是否进行了更改！仅供参考。
        '''常见的assign为max_iou_assigner,根据anchor与gt bbox的iou确定anchor target, assign返回的数据为一个数据类AssignResult,包含num_gts:gt bbox数量, assigned_gt_inds:anchor对应的label。-1：忽略0：负样本, 正数：gt bbox对应的index,max_overlaps:anchor与gt bbox的最大iou labels:pos bbox对应的label'''
        if self.assigner.__class__.__name__ !="ATSSAssigner" and self.assigner.__class__.__name__ !="rbox_SASAssigner" :
            # 既不是ATSS分配也不是SASA分配
            assign_result = self.assigner.assign(pred_instances,gt_instances,gt_instances_ignore)
        else:
            # 使用SASssigner分配目标
            assign_result = self.assigner.assign(pred_instances,num_level_anchors_inside,gt_instances,gt_instances_ignore)
        # No sampling is required except for RPN and
        # Guided Anchoring algorithms
        # 是否采样
        '''assign完就是sample了，未指定sample类型，直接使用PseudoSampler这个采样类，这个类
        直接将所有有效的anchors提取出来'''
        sampling_result = self.sampler.sample(assign_result, pred_instances,gt_instances)
        '''当有了anchor的target gt之后，还需要将bbox转换成delta,下面代码所做的是计算pos neg anchor对应的delta
        和权重赋值'''
        # 有效的anchor数量
        num_valid_anchors = anchors.shape[0]
        target_dim = gt_instances.bboxes.size(-1) if self.reg_decoded_bbox else self.bbox_coder.encode_size
        # CHANGED
        # bbox_gt = anchors.new_zeros(num_valid_anchors, target_dim)
        # bbox_targets:顾名思义，他是需要回归的bbox的目标！
        # 但是不能用它进行距离的衡量，因为正样本的bboxtargets是对应的GT标签，因此新建一个变量（bbox_anchors来进行相应的距离衡量！）
        # bbox_targets = torch.zeros_like(anchors)
        # bbox_anchors = torch.zeros_like(anchors)
        bbox_targets = anchors.new_zeros(num_valid_anchors, target_dim)
        # NEW2
        # bbox_gt_targets = anchors.new_zeros(num_valid_anchors, target_dim)

        # bbox_anchors = anchors.new_zeros(num_valid_anchors, target_dim)
        bbox_weights = anchors.new_zeros(num_valid_anchors, target_dim)

        # new
        angle_targets = anchors.new_zeros(num_valid_anchors, self.encode_size)
        angle_weights = anchors.new_zeros(num_valid_anchors, 1)


        # TODO: Considering saving memory, is it necessary to be long?
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # `bbox_coder.encode` accepts tensor or box type inputs and generates
        # tensor targets. If regressing decoded boxes, the code will convert
        # box type `pos_bbox_targets` to tensor.
        # NEW2
        distances = torch.zeros_like(anchors.tensor[:,-1]).reshape(-1)

        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_priors, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
                pos_bbox_targets = get_box_tensor(pos_bbox_targets)
            # bbox_gt[pos_inds, :] = pos_bbox_targets.float()
            bbox_targets[pos_inds, :] = pos_bbox_targets

            # NEW2
            # bbox_gt_targets[pos_inds, :] =
            gts=sampling_result.pos_gt_bboxes.regularize_boxes(self.bbox_coder.angle_version)
            gts=gts.float()
            priors=sampling_result.pos_priors
            priors = priors.tensor
            px, py, pw, ph, pt = priors.unbind(dim=-1)
            gx, gy, gw, gh, gt = gts.unbind(dim=-1)
            dx = (torch.cos(gt) * (px - gx) + torch.sin(gt) * (py - gy))
            dy = (-torch.sin(gt) * (px - gx) + torch.cos(gt) * (py - gy))
            # bbox_gt_targets[pos_inds, :] = (dx.pow(2) + dy.pow(2)).sqrt()
            distances[pos_inds]=(dx.pow(2)/ gw + dy.pow(2)/ gh).sqrt()
            # bbox_anchors[pos_inds, :] = get_box_tensor(anchors)[pos_inds, :]
            bbox_weights[pos_inds, :] = 1.0

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
            angle_targets = anchors.new_zeros(num_valid_anchors, 1)
            if self.use_encoded_angle:
                # Get encoded angle as target
                angle_targets[pos_inds, :] = pos_bbox_targets[:, 4:5]
            else:
                # Get gt angle as target
                angle_targets[pos_inds, :] = \
                    sampling_result.pos_gt_bboxes[:, 4:5]
            # Angle encoder
            angle_targets = self.angle_coder.encode(angle_targets)
            angle_weights[pos_inds, :] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # NEW!!!!!!
        # 动态评估（SA - M）核心代码实现 - -核心创新代码实现
        # 旋转框中心、宽度、高度、角度值
        # rbboxes_center, width, height, angles = bbox_targets[:, :2], bbox_targets[:, 2], bbox_targets[:, 3], bbox_targets[:,4]
        # anchor_center = bbox_anchors[:, :2]  # anchor中心
        # # 点相对于物体的距离，不分行列，改成一串
        # distances = torch.zeros_like(angles).reshape(-1)
        # # 角度属于0--二分之Π
        # angles_index_wh = ((width != 0) & (angles >= 0) & (angles <= 1.57)).squeeze()
        # # 否则使用hw_angles索引
        # angles_index_hw = ((width != 0) & ((angles < 0) | (angles > 1.57))).squeeze()
        # # 01_la:compution of distance--计算距离公式--0到Π/2
        # distances[angles_index_wh] = torch.sqrt(
        #     (torch.pow(rbboxes_center[angles_index_wh, 0] - anchor_center[angles_index_wh, 0], 2) / width[
        #         angles_index_wh].squeeze())
        #     + (torch.pow(rbboxes_center[angles_index_wh, 1] - anchor_center[angles_index_wh, 1], 2) / height[
        #         angles_index_wh].squeeze()))
        # # --计算距离公式 --否则
        # distances[angles_index_hw] = torch.sqrt(
        #     (torch.pow(rbboxes_center[angles_index_hw, 0] - anchor_center[angles_index_hw, 0], 2) / height[
        #         angles_index_hw].squeeze())
        #     + (torch.pow(rbboxes_center[angles_index_hw, 1] - anchor_center[angles_index_hw, 1], 2) / width[
        #         angles_index_hw].squeeze()))

        # distances = torch.zeros_like(bbox_gt_targets[-1]).reshape(-1)
        # distances = bbox_gt_targets.abs()[:,:2].sum(axis=1).sqrt()
        distances[distances.isnan()] = 0.

        # NEW 新版本距离公式的实现


        # sam权重=标签权重 * e^(-diatance)=e^(1/datance+1)加1防止除数为0,出错。这里权重计算和原文不一样
        sam_weights = label_weights * (torch.exp(1 / (distances + 1)))
        sam_weights[sam_weights.isinf()] = 0.
        sam_weights[sam_weights.isnan()] = 0.

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            # 多一个sam权重
            sam_weights = unmap(sam_weights, num_total_anchors, inside_flags)
            sam_weights=sam_weights.reshape(num_total_anchors,-1)
            bbox_weights=bbox_weights*sam_weights

            angle_targets = unmap(angle_targets, num_total_anchors,
                                  inside_flags)
            angle_weights = unmap(angle_weights, num_total_anchors,
                                  inside_flags)
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,neg_inds, sampling_result, angle_targets, angle_weights)

@MODELS.register_module()
class AngleBranchSAMS2ARefineHead(AngleBranchS2ARefineHead,AngleBranchSAMS2AHead):
    pass