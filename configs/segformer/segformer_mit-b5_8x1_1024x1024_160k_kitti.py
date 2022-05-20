_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/kitti_seg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b5.pth'),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

# optimizer

optimizer_config = dict()
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# optimizer

# runner = dict(type='IterBasedRunner', max_epochs=1000)
workflow = [('train', 2000), ('val', 1)]
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
checkpoint_config = dict(by_epoch=False, interval=4000)
data = dict(samples_per_gpu=4, workers_per_gpu=4)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook',
             init_kwargs={
                 'entity': 'ak6',
                 'project': 'mmseg_training_kitti_segFormer',
             },
             out_suffix=('.log', '.log.json', '.pth', '.py')
             ),
        # dict(type='TensorboardLoggerHook')
    ])

