_base_ = [
    '../_base_/models/anglebranchsasm.py',
    '../_base_/datasets/dataloader.py'
]
data_root = 'data/split_ms_dota/'
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file='trainval/annfiles/',
        data_prefix=dict(img_path='trainval/images/'),
    )
)
# val_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         ann_file='trainval/annfiles/',
#         data_prefix=dict(img_path='trainval/images/'),
#     )
# )
test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
    )
)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                    interval=1,
                    # save_best="dota/mAP",
                    ),#save
    # checkpoint=dict(type='CheckpointHook', interval=1,out_dir = './dota_dir'),
)
val_cfg=None
val_dataloader=None
val_evaluator=None