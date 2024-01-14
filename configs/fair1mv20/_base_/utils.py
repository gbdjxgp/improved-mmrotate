default_scope = 'mmrotate'
val_evaluator = dict(type='FAIR1MMetric', metric='mAP',collect_device='gpu')
# test_evaluator = val_evaluator
# test_evaluator = dict(
#     type='DOTAMetric',
#     metric='mAP',
#     outfile_prefix='./work_dirs/fair1m/s2anet_r50_fpn_fair1m')
test_evaluator = dict(
    type='FAIR1MMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/fair1mv20/Task1')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0/3,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[32, 44],
        gamma=0.1)
]
# optimizer = dict(
#     type='AdamW',
#     lr=0.0002 ,# /8*gpu_number,
#     betas=(0.9, 0.999),
#     weight_decay=0.05)
optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=35, norm_type=2)
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1000),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook',
                    # interval=1,
                    save_best="FAIR1M/mAP",
                    # out_dir='/media/e2112/62F4AE9B47D06C74/ckpt/',
                    ),#save
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook')
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='RotLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
angle_version = 'le90'
num_classes=37# 不加背景，仅计算类别