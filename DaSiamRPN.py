# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNotb
from run_SiamRPN import SiamRPN_init, SiamRPN_track
#from utils import get_axis_aligned_bbox, cxy_wh_2_rect

class DaSiamRPN():
    def __init__(self):
        # load net
        net = SiamRPNotb()
        net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNOTB.model')))
        net.eval().cuda()
        self.net = net

    def init(self, img, box):
        # image and init box
        #image_files = sorted(glob.glob('./bag/*.jpg'))
        #init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
        #[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

        x1, y1, x2, y2 = box
        cx, cy = (x1+x2)/2, (y1+y2)/2
        w, h = (x2-x1), (y2-y1)
        # tracker init
        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
        self.state = SiamRPN_init(img, target_pos, target_sz, self.net)

    def update(self, img):
        self.state = SiamRPN_track(self.state, img)  # track
        cx, cy = self.state['target_pos']
        w, h = self.state['target_sz']
        return np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2]), self.state['score']
