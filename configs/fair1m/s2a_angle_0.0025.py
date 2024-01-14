_base_ = './s2a_angle.py'
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))