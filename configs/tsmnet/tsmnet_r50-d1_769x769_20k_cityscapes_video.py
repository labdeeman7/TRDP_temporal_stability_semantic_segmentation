_base_ = [
    '../../_base_/models/tsm_r50-d8.py', '../../_base_/datasets/cityscapes_769x769.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_20k.py'
]

model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))