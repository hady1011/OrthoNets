# ADDED BY AUTHORS OF ORTHONET

_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_orthonet_ResConfig.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
