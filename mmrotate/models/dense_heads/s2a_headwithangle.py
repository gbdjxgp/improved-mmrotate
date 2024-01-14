# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import select_single_mlvl
from mmdet.utils import InstanceList, OptInstanceList,ConfigType,MultiConfig
from mmengine.config import ConfigDict
from torch import Tensor

from mmrotate.registry import MODELS,TASK_UTILS
from mmrotate.structures.bbox import RotatedBoxes
from mmdet.structures.bbox import get_box_tensor
from ..utils import ORConv2d, RotationInvariantPooling
from .rotated_retina_head import RotatedRetinaHead
# 代码照着AngleBranchRetinaHead改的。
from .angle_branch_retina_head import AngleBranchRetinaHead

@MODELS.register_module()
class S2AHeadWithAngle(AngleBranchRetinaHead):
    r"""An anchor-based head used in `S2A-Net
    <https://ieeexplore.ieee.org/document/9377550>`_.
    """  # noqa: W605

    def __init__(self,
                 *args,
                 use_encoded_angle: bool = True,
                 shield_reg_angle: bool = False,
                 use_normalized_angle_feat: bool = False,
                 angle_coder: ConfigType = dict(
                     type='CSLCoder',
                     angle_version='le90',
                     omega=1,
                     window='gaussian',
                     radius=6),
                 loss_angle: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(
                             type='Normal',
                             name='retina_cls',
                             std=0.01,
                             bias_prob=0.01),
                         dict(
                             type='Normal',
                             name='retina_angle_cls',
                             std=0.01,
                             bias_prob=0.01),
                     ]),
                 loss_bbox_type: str = 'normal',
                 **kwargs) -> None:
        self.angle_coder = TASK_UTILS.build(angle_coder)
        self.encode_size = self.angle_coder.encode_size
        super(AngleBranchRetinaHead,self).__init__(*args,init_cfg=init_cfg, **kwargs)
        self.loss_bbox_type = loss_bbox_type
        self.loss_angle = MODELS.build(loss_angle)
        self.shield_reg_angle = shield_reg_angle
        self.use_encoded_angle = use_encoded_angle
        self.use_normalized_angle_feat = use_normalized_angle_feat

    # s2a独有函数，自己改的
    def filter_bboxes(self, cls_scores: List[Tensor],
                      bbox_preds: List[Tensor],
                      angle_preds: List[Tensor],
                      ) -> List[List[Tensor]]:
        """This function will be used in S2ANet, whose num_anchors=1.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)

        Returns:
            list[list[Tensor]]: refined rbboxes of each level of each image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device=device)

        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 5)
            angle_pred = angle_preds[lvl]
            angle_pred = angle_pred.permute(0, 2, 3, 1)
            angle_pred = angle_pred.reshape(num_imgs, -1, self.angle_coder.encode_size)
            anchors = mlvl_anchors[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                angle_pred_i = angle_pred[img_id]
                angle_pred_i = self.angle_coder.decode(angle_pred_i)
                bbox_pred_i[:,4] = angle_pred_i
                decode_bbox_i = self.bbox_coder.decode(anchors, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list
    # 此函数继承rotatedretinanet和anglebranchretinanet
    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, anchors: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, bbox_weights: Tensor,
                            angle_targets: Tensor, angle_weights: Tensor,
                            avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            angle_pred (Tensor): Box angles for each scale
                level with shape (N, num_anchors * encode_size, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 5).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 5).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 5).
            angle_targets (Tensor): Angle regression targets of each anchor
                weight shape (N, num_total_anchors, 1).
            angle_weights (Tensor): Angle regression loss weights of each
                anchor with shape (N, num_total_anchors, 1).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # Equivalent substitution of ``@force_fp32()``
        cls_score = cls_score.float()
        bbox_pred = bbox_pred.float()
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        # 新加的
        # Shield angle in reg. branch
        if self.shield_reg_angle:
            bbox_weights[:, -1] = 0.
        # 新加的结束
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)
        if self.reg_decoded_bbox and (self.loss_bbox_type != 'kfiou'):
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)

        if self.loss_bbox_type == 'normal':
            loss_bbox = self.loss_bbox(
                bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        elif self.loss_bbox_type == 'kfiou':
            # When the regression loss (e.g. `KFLoss`)
            # is applied on both the delta and decoded boxes.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred_decode = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred_decode = get_box_tensor(bbox_pred_decode)
            bbox_targets_decode = self.bbox_coder.decode(anchors, bbox_targets)
            bbox_targets_decode = get_box_tensor(bbox_targets_decode)
            loss_bbox = self.loss_bbox(
                bbox_pred,
                bbox_targets,
                bbox_weights,
                pred_decode=bbox_pred_decode,
                targets_decode=bbox_targets_decode,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        # 新加的
        angle_pred = angle_pred.permute(0, 2, 3,
                                        1).reshape(-1, self.encode_size)
        angle_targets = angle_targets.reshape(-1, self.encode_size)
        angle_weights = angle_weights.reshape(-1, 1)

        loss_angle = self.loss_angle(
            angle_pred,
            angle_targets,
            weight=angle_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox,loss_angle

@MODELS.register_module()
class S2ARefineHeadWithAngle(AngleBranchRetinaHead):
    r"""Rotated Anchor-based refine head. It's a part of the Oriented Detection
    Module (ODM), which produces orientation-sensitive features for
    classification and orientation-invariant features for localization.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        frm_cfg (dict): Config of the feature refine module.
    """  # noqa: W605


    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 *args,
                 frm_cfg: dict = None,
                 use_encoded_angle: bool = True,
                 shield_reg_angle: bool = False,
                 use_normalized_angle_feat: bool = False,
                 angle_coder: ConfigType = dict(
                     type='CSLCoder',
                     angle_version='le90',
                     omega=1,
                     window='gaussian',
                     radius=6),
                 loss_angle: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(
                             type='Normal',
                             name='retina_cls',
                             std=0.01,
                             bias_prob=0.01),
                         dict(
                             type='Normal',
                             name='retina_angle_cls',
                             std=0.01,
                             bias_prob=0.01),
                     ]),
                 loss_bbox_type: str = 'normal',
                 **kwargs) -> None:
        self.angle_coder = TASK_UTILS.build(angle_coder)
        self.encode_size = self.angle_coder.encode_size
        # rotated
        super(AngleBranchRetinaHead,self).__init__(*args,num_classes=num_classes, in_channels=in_channels,init_cfg=init_cfg, **kwargs)
        self.loss_bbox_type = loss_bbox_type
        # anglebranch
        self.loss_angle = MODELS.build(loss_angle)
        self.shield_reg_angle = shield_reg_angle
        self.use_encoded_angle = use_encoded_angle
        self.use_normalized_angle_feat = use_normalized_angle_feat
        # s2arefine
        self.feat_refine_module = MODELS.build(frm_cfg)
        self.bboxes_as_anchors = None

    # initlayers重写
    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.or_conv = ORConv2d(
            self.in_channels,
            int(self.feat_channels / 8),
            kernel_size=3,
            padding=1,
            arf_config=(1, 8))
        self.or_pool = RotationInvariantPooling(self.feat_channels, 8)
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels / 8) if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        reg_dim = self.bbox_coder.encode_size
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * reg_dim, 3, padding=1)
        # anglebranch
        self.retina_angle_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.encode_size,
            3,
            padding=1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:

            - cls_score (Tensor): Cls scores for a single scale level
              the channels number is num_anchors * num_classes.
            - bbox_pred (Tensor): Box energies / deltas for a single scale
              level, the channels number is num_anchors * 5.
            - angle_pred (Tensor): Angle for a single scale level the channels
              number is num_anchors * encode_size.
        """

        x = self.or_conv(x)
        reg_feat = x
        cls_feat = self.or_pool(x)

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        angle_pred = self.retina_angle_cls(reg_feat)
        if self.use_normalized_angle_feat:
            angle_pred = angle_pred.sigmoid() * 2 - 1
        return cls_score, bbox_pred, angle_pred

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     angle_preds: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None,
                     rois: List[Tensor] = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            angle_preds (list[Tensor]): Box angles for each scale
                level with shape (N, num_anchors * encode_size, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            rois (list[Tensor])

        Returns:
            dict: A dictionary of loss components.
        """
        assert rois is not None
        self.bboxes_as_anchors = rois
        return super(S2ARefineHeadWithAngle, self).loss_by_feat(
            cls_scores=cls_scores,
            bbox_preds=bbox_preds,
            angle_preds=angle_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    batch_img_metas: List[dict],
                    device: Union[torch.device, str] = 'cuda') \
            -> Tuple[List[List[Tensor]], List[List[Tensor]]]:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            batch_img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors.
                Defaults to cuda.

        Returns:
            tuple:

            - anchor_list (list[list[Tensor]]): Anchors of each image.
            - valid_flag_list (list[list[Tensor]]): Valid flags of each
              image.
        """
        anchor_list = [[
            RotatedBoxes(bboxes_img_lvl).detach()
            for bboxes_img_lvl in bboxes_img
        ] for bboxes_img in self.bboxes_as_anchors]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        angle_preds: List[Tensor],
                        score_factors: Optional[List[Tensor]] = None,
                        rois: List[Tensor] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            angle_preds (list[Tensor]): Box angles for each scale
                level with shape (N, num_anchors * encode_size, H, W)
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            rois (list[Tensor]):
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        assert rois is not None

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        # mlvl_priors = self.prior_generator.grid_priors(
        #     featmap_sizes,
        #     dtype=cls_scores[0].dtype,
        #     device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(
                cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            angle_pred_list = select_single_mlvl(
                angle_preds, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                angle_pred_list=angle_pred_list,
                score_factor_list=score_factor_list,
                mlvl_priors=rois[img_id],
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def feature_refine(self, x: List[Tensor],
                       rois: List[List[Tensor]]) -> List[Tensor]:
        """Refine the input feature use feature refine module.

        Args:
            x (list[Tensor]): feature maps of multiple scales.
            rois (list[list[Tensor]]): input rbboxes of multiple
                scales of multiple images, output by former stages
                and are to be refined.

        Returns:
            list[Tensor]: refined feature maps of multiple scales.
        """
        return self.feat_refine_module(x, rois)

    def refine_bboxes(self, cls_scores: List[Tensor], bbox_preds: List[Tensor],
                      rois: List[List[Tensor]]) -> List[List[Tensor]]:
        """Refine predicted bounding boxes at each position of the feature
        maps. This method will be used in R3Det in refinement stages.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, 5, H, W)
            rois (list[list[Tensor]]): input rbboxes of each level of each
                image. rois output by former stages and are to be refined

        Returns:
            list[list[Tensor]]: best or refined rbboxes of each level of each
            image.
        """
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)

        num_imgs = cls_scores[0].size(0)

        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        bboxes_list = [[] for _ in range(num_imgs)]

        assert rois is not None
        mlvl_rois = [torch.cat(r) for r in zip(*rois)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            rois = mlvl_rois[lvl]
            assert bbox_pred.size(1) == 5
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(-1, 5)
            refined_bbox = self.bbox_coder.decode(rois, bbox_pred)
            refined_bbox = refined_bbox.reshape(num_imgs, -1, 5)
            for img_id in range(num_imgs):
                bboxes_list[img_id].append(refined_bbox[img_id].detach())
        return bboxes_list
