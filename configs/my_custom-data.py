_base_ = 'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

dataset_type = 'CocoDataset'
classes = ('balloon',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        img_prefix='data/balloon/train',
        classes=classes,
        ann_file='data/balloon/annotations/train/train.json'),
    val=dict(
        img_prefix='data/balloon/val',
        classes=classes,
        ann_file='data/balloon/annotations/val/val.json'),
    test=dict(
        img_prefix='data/balloon/val',
        classes=classes,
        ann_file='data/balloon/annotations/val/val.json'))
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'