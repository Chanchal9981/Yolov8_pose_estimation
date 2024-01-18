import argparse
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
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def output_to_keypoint(output):
    targets = []
    for i, o in enumerate(output):
        kpts = o[:,6:]
        o = o[:,:6]
        for index, (*box, conf, cls) in enumerate(o.cpu().numpy()):
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.cpu().numpy()[index])])
    return np.array(targets)


import cv2
input_size=640 ####add w of image or resize the image(640,640)
palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                        [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                        [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                        dtype=numpy.uint8)
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
model = torch.load("weights/best.pt", map_location='cpu')['model'].float()
stride = int(max(model.stride.cpu().numpy()))
model.eval()
data="images7"
j=0
for j in os.listdir(data):
    image=os.path.join(data,j)
    image=cv2.imread(image)
    # image=cv2.resize(image,(640,640)) ##resize the image same as input size
    # cv2.imwrite("ss1.jpg",image)
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
    image = numpy.ascontiguousarray(image)
    image = torch.from_numpy(image)
    image = image.unsqueeze(dim=0)
    image = image / 255
    outputs = model(image)
    outputs = util.non_max_suppression(outputs,0.4,0.4,model.head.nc)
    im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    im0 = im0.cpu().numpy().astype(np.uint8)
    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
    out=output_to_keypoint(outputs)
    for i, pose in enumerate(outputs):
        for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])):
            kpts = pose[det_index, 6:]
            kpts[:51:3] = kpts[:51:3]
            kpts[1:51:3] = kpts[1:51:3]
            # kpts[2::3] = 1
            # Convert tensor to list
            x1, y1, x2, y2 = xyxy
            x_min, y_min, x_max, y_max = int(x1), int(y1), int(x2), int(y2)
            print( x_min, y_min, x_max, y_max)
            kpts_lst = kpts.tolist()
            # print(kpts_lst)
                # kpts_lst = [int(x) for x in kpts_lst]
                # kpts_lst=' '.join(str(x) for x in kpts_lst)
            new_kp=[]
            for k in range(0, len(kpts) - 2, 3):
                new_kp.append(kpts_lst[k]/im0.shape[1])
                new_kp.append(kpts_lst[k+1]/im0.shape[0])
                new_kp.append(kpts_lst[k+2])
            istToStr = ' '.join([str(elem) for elem in new_kp])
            x_center = (x_min + x_max) / 2 / im0.shape[1]
            y_center = (y_min + y_max) / 2 / im0.shape[0]
            width = (x_max - x_min) / im0.shape[1]
            height = (y_max - y_min) / im0.shape[0]
            with open("images7/"+os.path.splitext(j)[0]+".txt",'a') as f:
                class_id = "0"
                f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {istToStr} \n')













# import argparse
# import copy
# import csv
# import os
# import warnings
# import math
# import numpy
# import torch
# import tqdm
# import yaml
# from torch.utils import data
# from nets import nn
# from utils import util
# from utils.dataset import Dataset
# import numpy as np
# import cv2
# import imutils

# def xyxy2xywh(x):
#     # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
#     y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#     y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
#     y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
#     y[:, 2] = x[:, 2] - x[:, 0]  # width
#     y[:, 3] = x[:, 3] - x[:, 1]  # height
#     return y

# def output_to_keypoint(output):
#     targets = []
#     for i, o in enumerate(output):
#         kpts = o[:,6:]
#         o = o[:,:6]
#         for index, (*box, conf, cls) in enumerate(o.cpu().numpy()):
#             targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf, *list(kpts.cpu().numpy()[index])])
#     return np.array(targets)


# import cv2
# input_size=640 ####add w of image or resize the image(640,640)
# palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
#                         [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
#                         [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                         [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
#                         dtype=numpy.uint8)
# skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
#             [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
# kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
# model = torch.load("yolov8s-pose.pt", map_location='cpu')['model'].float()
# stride = int(max(model.stride.cpu().numpy()))
# model.eval()
# image="ss1.png"
# image=cv2.imread(image)
# image=cv2.resize(image,(640,640))
# # cv2.imwrite("ss1.jpg",image)
# shape = image.shape[:2]  # current shape [height, width]
# r = min(1.0, input_size / shape[0], input_size / shape[1])
# pad = int(round(shape[1] * r)), int(round(shape[0] * r))
# w = input_size - pad[0]
# h = input_size - pad[1]
# w = numpy.mod(w, stride)
# h = numpy.mod(h, stride)
# w /= 2
# h /= 2
# if shape[::-1] != pad:
#     image = cv2.resize(image,
#     dsize=pad,
#     interpolation=cv2.INTER_LINEAR)
# top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
# left, right = int(round(w - 0.1)), int(round(w + 0.1))
# image = cv2.copyMakeBorder(image,
# top, bottom,
# left, right,
# cv2.BORDER_CONSTANT)  # add border
# image = image.transpose((2, 0, 1))[::-1]
# image = numpy.ascontiguousarray(image)
# image = torch.from_numpy(image)
# image = image.unsqueeze(dim=0)
# image = image / 255
# outputs = model(image)
# outputs = util.non_max_suppression(outputs,0.4,0.4,model.head.nc)
# out=output_to_keypoint(outputs)
# im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
# im0 = im0.cpu().numpy().astype(np.uint8)
# im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
# gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
# for i, pose in enumerate(outputs):
#     for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])):
#         kpts = pose[det_index, 6:]
#         kpts[:51:3] = kpts[:51:3]
#         kpts[1:51:3] = kpts[1:51:3]
#         x1, y1, x2, y2 = xyxy
#         x_min, y_min, x_max, y_max = int(x1), int(y1), int(x2), int(y2)
#         print( x_min, y_min, x_max, y_max)
#         kpts_lst = kpts.tolist()
#         with open("ss1.txt","a") as f:
#             f.write(f'{x_min} {y_min} {x_max} {y_max}' '\n')
# cv2.imwrite("ss1.jpg",im0)





        

