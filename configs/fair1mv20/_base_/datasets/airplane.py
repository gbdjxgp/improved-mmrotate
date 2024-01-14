_base_ = 'dataloader.py'
train_dataloader=dict(
    dataset=dict(
        ann_file='train/labelTxt_airplane/',
    )
)
val_dataloader=dict(
    dataset=dict(
        ann_file='validation/labelTxt_airplane/',
    )
)
