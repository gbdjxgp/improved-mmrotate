_base_ = 'baseline.py'
num_classes={{_base_.num_classes}}
angle_version={{_base_.angle_version}}
model = dict(
    bbox_head_init=dict(
        type='AngleBranchSAMS2AHead',
        use_normalized_angle_feat=True,
        shield_reg_angle=True,
        angle_coder=dict(
            type='PSCCoder',
            angle_version=angle_version,
            dual_freq=True,
            num_step=3),
        loss_angle=dict(type='mmdet.L1Loss', loss_weight=0.2)
    ),
    bbox_head_refine=[
        dict(
            type='AngleBranchSAMS2ARefineHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            frm_cfg=dict(
                type='AlignConv',
                feat_channels=256,
                kernel_size=3,
                strides=[8, 16, 32, 64, 128]),
            anchor_generator=dict(
                type='PseudoRotatedAnchorGenerator',
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
            use_normalized_angle_feat=True,
            shield_reg_angle=True,
            angle_coder = dict(
                type='PSCCoder',
                angle_version=angle_version,
                dual_freq=True,
                num_step=3),
            loss_angle=dict(type='mmdet.L1Loss', loss_weight=0.2)
        )
    ],
    train_cfg=dict(
        init=dict(
            _delete_=True,
            assigner=dict(
                type='rbox_SASAssigner',
                topk=9,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=[
            dict(
                assigner=dict(
                    type='mmdet.MaxIoUAssigner',
                    pos_iou_thr=0.6,#changed
                    neg_iou_thr=0.5,#changed
                    min_pos_iou=0,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type='RBboxOverlaps2D')),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1.0]
    ),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms_rotated', iou_threshold=0.1),
        max_per_img=2000
    )
)
