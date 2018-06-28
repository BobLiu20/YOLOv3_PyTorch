TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 80,
    },
    "batch_size": 16,
    "confidence_threshold": 0.5,
    "images_path": "./images/",
    "classes_names_path": "../data/coco.names",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    "pretrain_snapshot": "../weights/yolov3_weights_pytorch.pth",
}
