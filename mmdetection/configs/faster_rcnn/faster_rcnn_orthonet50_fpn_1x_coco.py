# ADDED BY AUTHORS OF ORTHONET

_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_orthonet.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
