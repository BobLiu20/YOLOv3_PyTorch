# coding='utf-8'
import os
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
import sys
import numpy as np
import time
import datetime
import importlib
import logging
import shutil

import matplotlib
matplotlib.use('Agg') #  disable display
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import torch.nn as nn


MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.coco_dataset import COCODataset
from common.utils import non_max_suppression


def evaluate(config):
    is_training = False
    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("Load checkpoint: {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        logging.warning("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # DataLoader.
    dataloader = torch.utils.data.DataLoader(COCODataset(config["val_path"],
                                                         (config["img_w"], config["img_h"]),
                                                         is_training=False),
                                             batch_size=config["batch_size"],
                                             shuffle=False, num_workers=8, pin_memory=False)

    # Coco Prepare.
    index2category = json.load(open("coco_index2category.json"))

    # Start the eval loop
    logging.info("Start eval.")
    coco_results = []
    coco_img_ids= set([])
    for step, samples in enumerate(dataloader):
        images, labels = samples["image"], samples["label"]
        image_paths, origin_sizes = samples["image_path"], samples["origin_size"]
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                                   conf_thres=0.01,
                                                   nms_thres=0.45)
        for idx, detections in enumerate(batch_detections):
            image_id = int(os.path.basename(image_paths[idx])[-16:-4])
            coco_img_ids.add(image_id)
            if detections is not None:
                origin_size = eval(origin_sizes[idx])
                detections = detections.cpu().numpy()
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                    x1 = x1 / config["img_w"] * origin_size[0]
                    x2 = x2 / config["img_w"] * origin_size[0]
                    y1 = y1 / config["img_h"] * origin_size[1]
                    y2 = y2 / config["img_h"] * origin_size[1]
                    w = x2 - x1
                    h = y2 - y1
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": index2category[str(int(cls_pred.item()))],
                        "bbox": (float(x1), float(y1), float(w), float(h)),
                        "score": float(conf),
                    })
        logging.info("Now {}/{}".format(step, len(dataloader)))
    save_results_path = "coco_results.json"
    with open(save_results_path, "w") as f:
        json.dump(coco_results, f, sort_keys=True, indent=4, separators=(',', ':'))
    logging.info("Save coco format results to {}".format(save_results_path))

    #  COCO api
    logging.info("Using coco-evaluate tools to evaluate.")
    cocoGt = COCO(config["annotation_path"])
    cocoDt = cocoGt.loadRes(save_results_path)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.imgIds  = list(coco_img_ids) # real imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python eval_coco.py params.py")
        sys.exit()
    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    evaluate(config)

if __name__ == "__main__":
    main()
