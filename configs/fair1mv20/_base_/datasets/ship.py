_base_ = 'dataloader.py'
train_dataloader=dict(
    dataset=dict(
        ann_file='train/labelTxt_ship/',
    )
)
val_dataloader=dict(
    dataset=dict(
        ann_file='validation/labelTxt_ship/',
    )
)