# ** Need to deal with bathcing.
#** TODO need to look at batching in pytorch to ensure images go in, the way I want. Also need to make a test
_base_ = './cityscapes.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (32, 32) #** Ensure that the images are around the size of 100x50 when I perform the downscaling. 
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(100, 50), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'sequence_imgs', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(100, 50),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='leftImg8bit/train',
#         ann_dir='gtFine_19/train',
#         sequence_dir='leftImg8bit_sequence/train',
#         sequence_range=2, #** figure out exactly how to use these.
#         sequence_num=2,
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='leftImg8bit/val',
#         ann_dir='gtFine_19/val',
#         sequence_dir='leftImg8bit_sequence/val',
#         sequence_range=2,
#         sequence_num=2,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='leftImg8bit',
#         ann_dir='gtFine_19',
#         sequence_dir='leftImg8bit_sequence',
#         split='val.txt',
#         sequence_range=2,
#         sequence_num=2,
#         pipeline=test_pipeline))
