# coding='utf-8'
import os
import sys
import numpy as np
import time
import datetime
import json
import importlib
import logging
import shutil

import torch
import torch.nn as nn


MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.coco_dataset import COCODataset
from common.utils import non_max_suppression, bbox_iou


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
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        logging.warning("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # DataLoader
    dataloader = torch.utils.data.DataLoader(COCODataset(config["val_path"]),
                                             batch_size=config["batch_size"],
                                             shuffle=False, num_workers=16, pin_memory=False)

    # Start the eval loop
    logging.info("Start eval.")
    n_gt = 0
    correct = 0
    for step, (images, labels) in enumerate(dataloader):
        labels = labels.cuda()
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            output = non_max_suppression(output, 80, conf_thres=0.2)
            #  calculate
            for sample_i in range(labels.size(0)):
                # Get labels for sample where width is not zero (dummies)
                target_sample = labels[sample_i, labels[sample_i, :, 3] != 0]
                for obj_cls, tx, ty, tw, th in target_sample:
                    # Get rescaled gt coordinates
                    tx1, tx2 = config["img_w"] * (tx - tw / 2), config["img_w"] * (tx + tw / 2)
                    ty1, ty2 = config["img_h"] * (ty - th / 2), config["img_h"] * (ty + th / 2)
                    n_gt += 1
                    box_gt = torch.cat([coord.unsqueeze(0) for coord in [tx1, ty1, tx2, ty2]]).view(1, -1)
                    sample_pred = output[sample_i]
                    if sample_pred is not None:
                        # Iterate through predictions where the class predicted is same as gt
                        for x1, y1, x2, y2, conf, obj_conf, obj_pred in sample_pred[sample_pred[:, 6] == obj_cls]:
                            box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)
                            iou = bbox_iou(box_pred, box_gt)
                            if iou >= config["iou_thres"]:
                                correct += 1
                                break
        if n_gt:
            logging.info('Batch [%d/%d] mAP: %.5f' % (step, len(dataloader), float(correct / n_gt)))

    logging.info('Mean Average Precision: %.5f' % float(correct / n_gt))

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python training.py params.py")
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
