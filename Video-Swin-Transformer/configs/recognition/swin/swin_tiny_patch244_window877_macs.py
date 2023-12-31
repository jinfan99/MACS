_base_ = [
    '../../_base_/models/swin/swin_tiny.py', '../../_base_/default_runtime.py'
]
model = dict(backbone=dict(patch_size=(2, 4, 4), drop_path_rate=0.1), test_cfg=dict(max_testing_views=4),
             cls_head=dict(num_classes=2,
                           loss_cls=dict(type='CrossEntropyLoss', class_weight=[0.2, 0.8])), )

DOWN_RATIO = 2

# dataset settings
dataset_type = 'RawframeDataset'
data_root_train = None
data_root_val = None
ann_file_train = None
ann_file_val = None
ann_file_test = None
img_norm_cfg = dict(
    mean=[136.074, 74.512, 62.077], std=[69.016, 61.919, 56.320], to_bgr=False)
# mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    # dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=int(32 / DOWN_RATIO),
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    # dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=int(32 / DOWN_RATIO),
        frame_interval=2,
        num_clips=4,
        test_mode=True),
    dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(-1, 224)),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root_train,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, betas=(0.9, 0.999), weight_decay=0.02,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    # warmup_iters=2.5
    warmup_iters=3
)
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/barr_swin_tiny_patch244_window877_macs'
find_unused_parameters = False

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=4,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    # use_fp16=True,
    use_fp16=False,
)
