from __future__ import print_function
import time
import sys
import os
import pickle
import argparse
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
#from data import VOCroot,COCOroot 
sys.path.append(".")
from data import AnnotationTransform, VOCDetection, BaseTransform,  COCO_mobile_300

import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer

#VOCroot = "./data/VOCdevkit/"

from models.RFB_Net_mobile import build_net

from collections import OrderedDict

class RFBNetDetector():
    def __init__(self): 
        cfg = COCO_mobile_300
        priorbox = PriorBox(cfg)
        with torch.no_grad():
            priors = priorbox.forward()
            priors = priors.cuda()
        # load net
        img_dim = 300
        num_classes = 4
        net = build_net('test', img_dim, num_classes)    # initialize detector
        state_dict = torch.load("./weights/RFB_mobile_VOC_epoches_230.pth")
        # create new OrderedDict that does not contain `module.`
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        net.eval()
        print('Finished loading model!')
        #print(net)
        #testset = VOCDetection(VOCroot, [('2019', 'test')], None, AnnotationTransform())
        #img = cv2.imread("demo.jpg", cv2.IMREAD_COLOR)
        #imgs = [img]*10
        
        net = net.cuda()
        cudnn.benchmark = True

        # evaluation
        #top_k = (300, 200)[args.dataset == 'COCO']
        top_k = 200
        detector = Detect(num_classes,0,cfg)
        rgb_means = (103.94,116.78,123.68)
        #test_net(net, detector, imgs,
        #         BaseTransform(net.size, rgb_means, (2, 0, 1)),
        #         top_k, thresh=0.01)
        self.net = net
        self.detector = detector
        self.transform = BaseTransform(net.size, rgb_means, (2, 0, 1))
        self.priors = priors
        self._t = {'im_detect': Timer(), 'misc': Timer()}
        
    def detect(self, img, thresh = 0.1):
        #print("1: {}".format(time.time()))
        net, detector, transform, priors = self.net, self.detector, self.transform, self.priors
        _t = self._t
        max_per_image=100
        num_images = 1
        num_classes = 3
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(num_classes)]
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            x = x.cuda()
            scale = scale.cuda()

        #print("2: {}".format(time.time()))
        _t['im_detect'].tic()
        with torch.no_grad():
            out = net(x)      # forward pass
            boxes, scores = detector.forward(out,priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image

        _t['misc'].tic()
        
        i=0
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=False)
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()
        
        #print(all_boxes[j][i])
        '''
        color = None
        rect = None
        for j in range(1, num_classes):
            boxes = np.array(all_boxes[j][i])
            if len(boxes) == 0:
                continue
            sort_arg = np.argsort(boxes[:,-1])
            if boxes[sort_arg[-1],-1] < 0.1:
                continue
            rect = [int(i) for i in boxes[sort_arg[-1],:4]]
            color = ["blue", "red"][j == 2]
        

        if i % 100 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()
            
        '''
        return all_boxes


if __name__ == '__main__':
    detector = RFBNetDetector()
    #cap = cv2.VideoCapture("/home/zhikun/Videos/shaobing/*.jpg")
    #ret, frame = cap.read()
    x = 1565439323
    while True:
        #ret, frame = cap.read()
        
        frame = cv2.imread("/home/zhikun/Videos/shaobing/"+str(x)+".jpg")
        x += 1
        frame_to_save = frame.copy()
        boxes = detector.detect(frame)
        blue_boxes = np.array(boxes[1][0])    
        if blue_boxes.size == 0:
            armor_box_blue = None
        else:
            confidence = blue_boxes[:, -1]
            max_arg = np.argmax(confidence)
            armor_box_blue = blue_boxes[max_arg, :4]
        if armor_box_blue is not None:
            x1, y1, x2, y2 = armor_box_blue
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2),(255, 0, 0), 2)
            
        red_boxes = np.array(boxes[2][0])
        if red_boxes.size == 0:
            armor_box_red = None
        else:
            confidence = red_boxes[:, -1]
            max_arg = np.argmax(confidence)
            armor_box_red = red_boxes[max_arg, :4]
        if armor_box_red is not None:
            x1, y1, x2, y2 = armor_box_red
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 0, 255), 2)
         
        cnt = 0
        cv2.imshow("frame", frame)
        cv2.waitKey(0)

    
