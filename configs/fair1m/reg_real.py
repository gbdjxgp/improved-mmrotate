# _base_ = 's2anet_r50_fpn_5x_fair1m.py'
# _base_ = \
#     ['../rotated_retinanet/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py']
#
# angle_version = 'le90'
# model = dict(
#     bbox_head=dict(
#         anchor_generator=dict(angle_version=None),
#         type='AngleBranchRetinaHead',
#         use_normalized_angle_feat=True,
#         angle_coder=dict(
#             type='PSCCoder',
#             angle_version=angle_version,
#             dual_freq=True,
#             num_step=3),
#         loss_cls=dict(
#             type='mmdet.FocalLoss',
#             use_sigmoid=True,
#             gamma=2.0,
#             alpha=0.25,
#             loss_weight=1.0),
#         loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.5),
#         loss_angle=dict(type='mmdet.L1Loss', loss_weight=0.2)))

_base_ = 's2anet_r50_fpn_5x_fair1m.py'
angle_version='le90'
default_hooks = dict(
checkpoint=dict(type='CheckpointHook', interval=1,save_best="dota/mAP",out_dir='/media/e2112/62F4AE9B47D06C74/ckpt/',),#save
)
model = dict(
    bbox_head_init=dict(
        type='S2AHeadWithAngle',
        anchor_generator=dict(
            angle_version=angle_version, ),
        use_normalized_angle_feat=True,
        shield_reg_angle=True,
        bbox_coder=dict(
            angle_version=angle_version, ),
        angle_coder = dict(
            type='PSCCoder',
            angle_version=angle_version,
            dual_freq=True,
            num_step=3),
        loss_angle=dict(type='mmdet.L1Loss', loss_weight=0.2)

    ),
    bbox_head_refine=[
        dict(
            type='S2ARefineHeadWithAngle',
            num_classes=37,
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



#
# bbox_head = dict(
# use_hbbox_loss = True,
# scale_angle = False,

# loss_cls = dict(
#     type='mmdet.FocalLoss',
#     use_sigmoid=True,
#     gamma=2.0,
#     alpha=0.25,
#     loss_weight=1.0),
# loss_bbox = dict(type='mmdet.IoULoss', loss_weight=1.0),
# loss_centerness = dict(
#     type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
# loss_angle = dict(_delete_=True, type='mmdet.L1Loss', loss_weight=0.2),)