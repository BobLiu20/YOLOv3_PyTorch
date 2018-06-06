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
    "lr": {
        "backbone_lr": 0.01,
        "other_lr": 0.01,
        "freeze_backbone": False,
        "decay_gamma": 0.01,
        "decay_step": 30,
    },
    "optimizer": {
        "type": "sgd",
        "weight_decay": 4e-05,
    },
    "batch_size": 16,
    "train_path": "/home/liubofang/bob/YOLOv3_PyTorch/data/coco/trainvalno5k.txt",
    "epochs": 100,
    "img_h": 416,
    "img_w": 416,
    "parallels": [4,5,6,7],
    "working_dir": "/world/data-c9/liubofang/training/yolo3/pytorch",
    "pretrain_snapshot": "",
    "evaluate_type": "", 
    "try": 1060,
    "export_onnx": False,
}
