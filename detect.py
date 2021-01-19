import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords,
    xyxy2xywh, set_logging)
from utils.torch_utils import select_device, time_synchronized

# Original source code: https://github.com/ultralytics/yolov5
# Customized by Hokwang Choi.
class DetectorYoloV5:
    def __init__(self, conf_thres, iou_thres):
        # The original image size of the trained network.
        self.imgsz = 640
        self.weights = 'yolov5s.pt'
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Initialize the device.
        set_logging()
        # GPU device 0 by default.
        self.device = select_device('')
        # Half precision only supported on CUDA.
        self.half = self.device.type != 'cpu'  

        # Load FP32 model.
        self.model = attempt_load(self.weights, map_location=self.device)
        # Check img_size.
        self.checked_imgsz = check_img_size(self.imgsz, s=self.model.stride.max())
        if self.half:
            self.model.half()  # to FP16

        # Get class names.
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Dry run to prepare GPU.
        img = torch.zeros((1, 3, self.checked_imgsz, self.checked_imgsz), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None


    def detect(self, sub_image = None):
        img_raw = sub_image # BGR
        assert img_raw is not None, 'Image is not valid.'
        
        # Padded resize.
        img_test = self.letterbox(img_raw, new_shape=self.checked_imgsz)[0]

        # Convert image to fit the input layer.
        img_test = img_test[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_test = np.ascontiguousarray(img_test)
        img_test = torch.from_numpy(img_test).to(self.device)
        img_test = img_test.half() if self.half else img_test.float()  # uint8 to fp16/32
        img_test /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_test.ndimension() == 3:
            img_test = img_test.unsqueeze(0)

        # Inference.
        t1 = time_synchronized()
        pred = self.model(img_test, augment=False)[0]

        # Apply NMS.
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None) # filtering by class possible
        t2 = time_synchronized()

        # Result format.
        result = []

        for i, det in enumerate(pred):
            s, im0 = '', img_raw
            s += 'Original image size: ' + '%gx%g ' % im0.shape[0:2]
            s += ', Resized for the input layer: ' + '%gx%g ' % img_test.shape[2:]
            s += ', Detection result: '

            if det is not None and len(det):
                # Rescale boxes from input layer size to original image size.
                det[:, :4] = scale_coords(img_test.shape[2:], det[:, :4], im0.shape).round()

                # Print results.
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results.
                for *xyxy, conf, clss in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    result += [[clss.tolist(), conf.tolist(), *xywh]]  # output format list of list
            
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
        return np.array(result)


    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
