# 最新
# 数据集

dataset_type = 'DOTADataset'
# data_root = 'data/FAIR1M2.0/'
data_root = 'data/split_ss_dota/'
backend_args = None
batch_size=2
num_workers=2
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        # ann_file='train-mini/annfiles/',
        # data_prefix=dict(img_path='train-mini/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(
                type='mmdet.RandomFlip',
                prob=0.5,
                direction=['horizontal', 'vertical', 'diagonal']),
            dict(type='RandomRotate',
                 prob=0.5,
                 angle_range=180,),
            dict(type='mmdet.Pad', size_divisor=32,),
            dict(type='mmdet.PackDetInputs'),
        ]
    )
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        # ann_file='val-mini/annfiles/',
        # data_prefix=dict(img_path='val-mini/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(
                type='mmdet.RandomFlip',
                prob=0.5,
                direction=['horizontal', 'vertical', 'diagonal']),
            # dict(type='RandomRotate',
            #      prob=0.5,
            #      angle_range=180,),
            dict(type='mmdet.Pad', size_divisor=32,),
            # dict(type='ImageToTensor', keys=['img']),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'
                           )
            )
        ]
    )
)




test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=None),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
# inference on test dataset and format the output results
# for submission. Note: the test set has no annotation.
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/images/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='DOTAMetric', metric='mAP',collect_device='gpu')
test_evaluator = dict(type='DOTAMetric', metric='mAP',collect_device='gpu')

test_evaluator = dict(
    type='DOTAMetric',
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/dota/Task1')