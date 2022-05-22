_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/kitti_seg_basic.py',
    '../_base_/default_runtime.py','../_base_/schedules/schedule_160k.py',
    '../_base_/wandb_logger_mmseg_training_kitti_segFormer.py'
]

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/mit_b0.pth')),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))

# optimizer

optimizer_config = dict()
optimizer = dict(
    _delete_=True,
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

val_interval = 800

# runner = dict(
#     _delete_=True,
#     type='EpochBasedRunner', max_epochs=100)
workflow = [('train', val_interval), ('val', 1)]
evaluation = dict(interval=val_interval, metric='mIoU', pre_eval=True, save_best='mIoU')

# runner = dict(type='EpochBasedRunner', max_epochs=100)
# workflow = [('train', 2000), ('val', 1)]
# evaluation = dict(interval=4000, metric='mIoU', pre_eval=True, save_best='mIoU')
# checkpoint_config = dict(by_epoch=False, interval=4000)
data = dict(samples_per_gpu=2, workers_per_gpu=4)


log_config={{_base_.customized_log_config}}

