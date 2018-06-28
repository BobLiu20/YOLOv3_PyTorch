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
import cv2
import random

import torch
import torch.nn as nn


MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou


def test(config):
    is_training = False
    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)

    # Set data parallel
    net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    if config["pretrain_snapshot"]:
        logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
        state_dict = torch.load(config["pretrain_snapshot"])
        net.load_state_dict(state_dict)
    else:
        raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
                                    config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # prepare images path
    images_name = os.listdir(config["images_path"])
    images_path = [os.path.join(config["images_path"], name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["images_path"]))

    # Start testing FPS of different batch size
    for batch_size in range(1, 10):
        # preprocess
        images = []
        for path in images_path[: batch_size]:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (config["img_w"], config["img_h"]),
                               interpolation=cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image /= 255.0
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            images.append(image)
        for i in range(batch_size-len(images)):
            images.append(images[0]) #  fill len to batch_sze
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        # inference in 30 times and calculate average
        inference_times = []
        for i in range(30):
            start_time = time.time()
            with torch.no_grad():
                outputs = net(images)
                output_list = []
                for i in range(3):
                    output_list.append(yolo_losses[i](outputs[i]))
                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, config["yolo"]["classes"],
                                                       conf_thres=config["confidence_threshold"])
                torch.cuda.synchronize() #  wait all done.
            end_time = time.time()
            inference_times.append(end_time - start_time)
        inference_time = sum(inference_times) / len(inference_times) / batch_size
        fps = 1.0 / inference_time
        logging.info("Batch_Size: {}, Inference_Time: {:.5f} s/image, FPS: {}".format(batch_size,
                                                                                     inference_time,
                                                                                     fps))



def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(asctime)s %(filename)s] %(message)s")

    if len(sys.argv) != 2:
        logging.error("Usage: python test_images.py params.py")
        sys.exit()
    params_path = sys.argv[1]
    if not os.path.isfile(params_path):
        logging.error("no params file found! path: {}".format(params_path))
        sys.exit()
    config = importlib.import_module(params_path[:-3]).TRAINING_PARAMS
    config["batch_size"] *= len(config["parallels"])

    # Start training
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config["parallels"]))
    test(config)


if __name__ == "__main__":
    main()
