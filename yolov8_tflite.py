import argparse
import tensorflow as tf
import os, sys
import cv2 as cv
import numpy
from utils import util
import torch
import time
from nets import nn
import copy
import csv
import os
import warnings
import math
import numpy
import torch
import tqdm
import yaml
from torch.utils import data
from nets import nn
from utils import util
from utils.dataset import Dataset
import numpy as np
import cv2
import imutils

def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  
    y[:, 2] = x[:, 2] - x[:, 0]  
    y[:, 3] = x[:, 3] - x[:, 1]  
    return y

def output_to_keypoint(output):
    targets = []
    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6]
        for index, (*box, conf, cls) in enumerate(o.cpu().numpy()):
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.cpu().numpy()[index])])
    return np.array(targets)


def fall_detection(poses):
    for pose in poses:
        xmin, ymin = (pose[2] - pose[4] / 2), (pose[3] - pose[5] / 2)
        xmax, ymax = (pose[2] + pose[4] / 2), (pose[3] + pose[5] / 2)
        left_shoulder_y = pose[23]
        left_shoulder_x = pose[22]
        right_shoulder_y = pose[26]
        left_body_y = pose[41]
        left_body_x = pose[40]
        right_body_y = pose[44]
        len_factor = math.sqrt(((left_shoulder_y - left_body_y) ** 2 + (left_shoulder_x - left_body_x) ** 2))
        left_foot_y = pose[53]
        right_foot_y = pose[56]
        dx = int(xmax) - int(xmin)
        dy = int(ymax) - int(ymin)
        difference = dy - dx
        if left_shoulder_y > left_foot_y - len_factor and left_body_y > left_foot_y - (
                len_factor / 2) and left_shoulder_y > left_body_y - (len_factor / 2) or (
                right_shoulder_y > right_foot_y - len_factor and right_body_y > right_foot_y - (
                len_factor / 2) and right_shoulder_y > right_body_y - (len_factor / 2)) \
                or difference < 0:
            return True, (xmin, ymin, xmax, ymax)
    return False, None

def falling_alarm(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    # cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
    #               thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(image, 'Fall', (90,150), 0,5, [0, 0, 255], thickness=5, lineType=cv2.LINE_AA)
    # cv2.putText(image,(150,120),cv2.FONT_HERSHEY_COMPLEX,0.7,((0,255,0)),1)


@torch.no_grad()
def demo(args):
    import cv2
    size = (17,3)
    stride = 32
    palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                           [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                           [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                           [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                          dtype=numpy.uint8)
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
    kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    input_details = interpreter.get_input_details()
    print("input",len(input_details))
    output_details = interpreter.get_output_details()
    print("output",len(output_details))
    height, width = input_details[0]["shape"][2], input_details[0]["shape"][3]
    print(height)
    interpreter.allocate_tensors()
    
    path="video_4.mp4"
    camera = cv2.VideoCapture(path)
    width_w = int(camera.get(3))
    height_h = int(camera.get(4))
    save = cv2.VideoWriter("key.mp4",cv.VideoWriter_fourcc(*'mp4v'),30,(width_w,height_h))
    input_s = height
    print(">>>",input_s)
    if not camera.isOpened():
        print("Error opening video stream or file")
    while camera.isOpened():
        success, frame = camera.read()
        if success:
            image = frame.copy()
            print(image.shape)
            image=cv2.resize(image,(640,640))
            frame=image
            shape = image.shape[:2]  # current shape [height, width]
            r = min(1.0, input_size / shape[0], input_size / shape[1])
            pad = int(round(shape[1] * r)), int(round(shape[0] * r))
            w = input_size - pad[0]
            h = input_size - pad[1]
            w = numpy.mod(w, stride)
            h = numpy.mod(h, stride)
            w /= 2
            h /= 2
            if shape[::-1] != pad:  # resize
                image = cv2.resize(image,
                                   dsize=pad,
                                   interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image,
                                       top, bottom,
                                       left, right,
                                       cv2.BORDER_CONSTANT)  # add border
            image = image.transpose((2, 0, 1))[::-1]
            print("***",image.shape)
            image = numpy.ascontiguousarray(image)
            image = torch.from_numpy(image)
            image = image.unsqueeze(dim=0)
            image = image / 255
            print("ima.extand",image.shape)
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            output_data1 = interpreter.get_tensor(output_details[0]['index'])
            output_data1 = torch.tensor(output_data1)
            outputs = util.non_max_suppression(output_data1,0.25,0.4,1)
            out=output_to_keypoint(outputs)
            # image=image.detach().numpy()
            for output in outputs:
                output = output.clone()
                is_fall,bbox=fall_detection(out)
                if is_fall:
                    falling_alarm(frame,bbox)
                boxes = output[:,:6]
                boxes = reversed(boxes)
                if len(output):
                    size=(17,3)
                    output = output[:, 6:].view(len(output),*size)
                    print(output.shape)
                else:
                    output = output[:,6:]
                r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])
                output[..., 0] -= (image.shape[3] - shape[1] * r) / 2  # x padding
                output[..., 1] -= (image.shape[2] - shape[0] * r) / 2  # y padding
                output[..., 0] /= r
                output[..., 1] /= r
                output[..., 0].clamp_(0, shape[1])  # x
                output[..., 1].clamp_(0, shape[0])  # y
                for kpt in reversed(output):
                    for i, k in enumerate(kpt):
                        color_k = [int(x) for x in kpt_color[i]]
                        x_coord, y_coord = k[0], k[1]
                        if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                            if len(k) == 3:
                                conf = k[2]
                                if conf < 0.5:
                                    continue
                            cv2.circle(frame,
                                       (int(x_coord), int(y_coord)),
                                       1, color_k, -1, lineType=cv2.LINE_AA)

                    for i, sk in enumerate(skeleton):
                        pos1 = (int(kpt[(sk[0] - 1), 0]), int(kpt[(sk[0] - 1), 1]))
                        pos2 = (int(kpt[(sk[1] - 1), 0]), int(kpt[(sk[1] - 1), 1]))
                        if kpt.shape[-1] == 3:
                            conf1 = kpt[(sk[0] - 1), 2]
                            conf2 = kpt[(sk[1] - 1), 2]
                            if conf1 < 0.5 or conf2 < 0.5:
                                continue
                        if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                            continue
                        if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                            continue
                        cv2.line(frame,
                                 pos1, pos2,
                                 [int(x) for x in limb_color[i]],
                                 thickness=2, lineType=cv2.LINE_AA)
                        # if i < len(boxes):
                        #     cv2.rectangle(frame,(int(boxes[i][0]), int(boxes[i][1])),(int(boxes[i][2]),int(boxes[i][3])),(255,0,0),2)
            # frame= imutils.resize(frame, width=1280)
            cv2.imshow('Frame',frame)
            save.write(frame) 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    camera.release()
    save.release()
if __name__ == "__main__":
    input_size = 640
    demo(input_size)
