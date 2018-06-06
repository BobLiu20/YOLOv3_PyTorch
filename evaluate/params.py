TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        # "backbone_imagenet_pretrain": False,
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "batch_size": 1,
    "iou_thres": 0.5,
    "val_path": "/home/liubofang/bob/YOLOv3_PyTorch/data/coco/5k.txt",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    # "pretrain_snapshot": "../weights/model.pth",
    "pretrain_snapshot": "/world/data-c9/liubofang/training/yolo3/pytorch/darknet_53/size416x416_try1060/20180605234635/model.pth",
}
