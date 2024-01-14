_base_ = 'baseline.py'
num_classes={{_base_.num_classes}}
angle_version={{_base_.angle_version}}
model = dict(
    bbox_head_init=dict(
        type='S2AHeadWithAngle',
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
            type='S2ARefineHeadWithAngle',
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
)