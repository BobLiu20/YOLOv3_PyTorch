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
import onnx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
import torch.nn as nn


MY_DIRNAME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(MY_DIRNAME, '..'))
from nets.model_main import ModelMain
from nets.yolo_loss import YOLOLoss
from common.utils import non_max_suppression, bbox_iou

cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

def resize_square(img, height=416, color=(0, 0, 0)):  # resize a rectangular image to a padded square
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = [round(shape[0] * ratio), round(shape[1] * ratio)]
    dw = height - new_shape[1]  # width padding
    dh = height - new_shape[0]  # height padding
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_AREA)  # resized, no border
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color), ratio, dw // 2, dh // 2

def plot_one_box(x, img, color, label=None, line_thickness=None):  # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1]), 0, tl / 5, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def test(config):
    is_training = False
    # Load and initialize network
    net = ModelMain(config, is_training=is_training)
    net.train(is_training)
    net.load_darknet_weights('../weights/pt_to_darknet.weight')
    torch.save(net.state_dict(), '../weights/darknet_to_pt.pt')

    # checkpoint = torch.load('../weights/darknet_to_pt.pt', map_location='cpu')
    # net.load_state_dict(checkpoint)

    torch.onnx.export(net, torch.from_numpy(np.ones([3,416,416],np.float32)).unsqueeze(0), '../weights/line_detector.onnx',
                      verbose=True,opset_version=9)
    # Set data parallel
    # net = nn.DataParallel(net)
    net = net.cuda()

    # Restore pretrain model
    # if config["pretrain_snapshot"]:
    #     logging.info("load checkpoint from {}".format(config["pretrain_snapshot"]))
    #     state_dict = torch.load(config["pretrain_snapshot"])
    #     net.load_state_dict(state_dict)
    # else:
    #     raise Exception("missing pretrain_snapshot!!!")

    # YOLO loss with 3 scales
    # yolo_losses = []
    # for i in range(3):
    #     yolo_losses.append(YOLOLoss(config["yolo"]["anchors"][i],
    #                                 config["yolo"]["classes"], (config["img_w"], config["img_h"])))

    # prepare images path
    images_name = os.listdir(config["images_path"])
    images_path = [os.path.join(config["images_path"], name) for name in images_name]
    if len(images_path) == 0:
        raise Exception("no image found in {}".format(config["images_path"]))

    # Start inference
    batch_size = 1
    for step in range(0, len(images_path), batch_size):
        # preprocess
        images = []
        images_origin = []
        for path in images_path[step*batch_size: (step+1)*batch_size]:
            logging.info("processing: {}".format(path))
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            images_origin.append(image)
            if image is None:
                logging.error("read path error: {}. skip it.".format(path))
                continue

            # Padded resize
            img, _, _, _ = resize_square(image, height=416, color=(127.5, 127.5, 127.5))

            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            # img -= self.rgb_mean
            # img /= self.rgb_std
            img /= 255.0
            images.append(img)
        images = np.asarray(images)
        images = torch.from_numpy(images).cuda()
        # inference
        with torch.no_grad():
            pred = net(images)
            # trace_model = torch.jit.trace(net,images)
            # pred_out = trace_model(images)
            # pred_out1 = trace_model(torch.from_numpy(np.ones([3,416,416],np.float32)).unsqueeze(0).cuda())
            # # trace_torch_model = False
            # trace_model.save('../weights/EVTraffic_LineDetector_v1.0.0.pt')
            # output_list = []
            # for i in range(3):
            #     output_list.append(yolo_losses[i](outputs[i]))
            # output = torch.cat(output_list, 1)
            pred = pred[pred[:, :, 4] > 0.2]
            print(pred)

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
