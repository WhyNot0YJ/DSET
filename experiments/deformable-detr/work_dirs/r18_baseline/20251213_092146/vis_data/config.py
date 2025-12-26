auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
custom_hooks = [
    dict(
        min_delta=0.0001,
        monitor='coco/bbox_mAP',
        patience=20,
        rule='greater',
        type='EarlyStoppingHook'),
]
data_root = '/root/autodl-tmp/datasets/DAIR-V2X/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 50
model = dict(
    as_two_stage=False,
    backbone=dict(
        depth=18,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet18', type='Pretrained'),
        norm_cfg=dict(requires_grad=False, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=8,
        sync_cls_avg_factor=True,
        type='DeformableDETRHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(batch_first=True, embed_dims=256),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1),
            self_attn_cfg=dict(
                batch_first=True, dropout=0.1, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.1),
            self_attn_cfg=dict(batch_first=True, embed_dims=256)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            128,
            256,
            512,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_feature_levels=4,
    num_queries=100,
    positional_encoding=dict(normalize=True, num_feats=128, offset=-0.5),
    test_cfg=dict(max_per_img=100),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DeformableDETR',
    with_box_refine=False)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            reference_points=dict(lr_mult=0.1),
            sampling_offsets=dict(lr_mult=0.1))),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=200,
        gamma=0.1,
        milestones=[
            160,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/root/autodl-tmp/datasets/DAIR-V2X/',
        metainfo=dict(
            classes=(
                'Car',
                'Truck',
                'Van',
                'Bus',
                'Pedestrian',
                'Cyclist',
                'Motorcyclist',
                'Trafficcone',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1280,
                720,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/root/autodl-tmp/datasets/DAIR-V2X/annotations/instances_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    metric_items=[
        'mAP',
        'mAP_50',
        'mAP_75',
        'mAP_s',
        'mAP_m',
        'mAP_l',
    ],
    proposal_nums=(
        1,
        10,
        100,
    ),
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=200, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_train.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/root/autodl-tmp/datasets/DAIR-V2X/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=(
                'Car',
                'Truck',
                'Van',
                'Bus',
                'Pedestrian',
                'Cyclist',
                'Motorcyclist',
                'Trafficcone',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PhotoMetricDistortion'),
            dict(min_crop_size=0.1, type='MinIoURandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                keep_ratio=True,
                scales=[
                    (
                        1280,
                        480,
                    ),
                    (
                        1280,
                        512,
                    ),
                    (
                        1280,
                        544,
                    ),
                    (
                        1280,
                        576,
                    ),
                    (
                        1280,
                        608,
                    ),
                    (
                        1280,
                        640,
                    ),
                    (
                        1280,
                        672,
                    ),
                    (
                        1280,
                        704,
                    ),
                    (
                        1280,
                        736,
                    ),
                    (
                        1280,
                        768,
                    ),
                    (
                        1280,
                        800,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='/root/autodl-tmp/datasets/DAIR-V2X/',
        metainfo=dict(
            classes=(
                'Car',
                'Truck',
                'Van',
                'Bus',
                'Pedestrian',
                'Cyclist',
                'Motorcyclist',
                'Trafficcone',
            )),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1280,
                720,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/root/autodl-tmp/datasets/DAIR-V2X/annotations/instances_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    metric_items=[
        'mAP',
        'mAP_50',
        'mAP_75',
        'mAP_s',
        'mAP_m',
        'mAP_l',
    ],
    proposal_nums=(
        1,
        10,
        100,
    ),
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/r18_baseline'
