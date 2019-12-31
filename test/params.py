TRAINING_PARAMS = \
{
    "model_params": {
        "backbone_name": "darknet_53",
        "backbone_pretrained": "",
    },
    "yolo": {
        "anchors": [[[0.52,300], [1.57,350], [2.62,400]],
                    [[0.52,105], [1.57,175], [2.62,210]],
                    [[0.52,18], [1.57,35], [2.62,52]]],
        "classes": 1,
    },
    "batch_size": 1,
    "confidence_threshold": 0.2,
    "images_path": "/media/hzh/work/workspace/yolov3-line_detect/test1",
    "classes_names_path": "../data/coco.names",
    "img_h": 416,
    "img_w": 416,
    "parallels": [0],
    "pretrain_snapshot": "../weights/official_yolov3_weights_pytorch.pth",
}
