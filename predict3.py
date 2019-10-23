import time
import math
import threading
import multiprocessing
from collections import namedtuple

import cv2
import numpy as np
from scipy.stats import linregress

from uart import Uart


class Memory():
    def __init__(self, max_size=3):
        self.max_size = max_size
        self.size = 0
        self.memory = np.zeros(self.max_size)
        self.full = False

    def put(self, x):
        self.memory[self.size] = x
        self.size += 1
        if self.size >= self.max_size:
            self.size = 0
            self.full = True

    def getAll(self):
        zero_to_now = self.memory[:self.size]
        older = self.memory[self.size:]
        return np.concatenate([older, zero_to_now], axis=0)

    def clean(self):
        self.size = 0
        self.full = False


class LinearPredictor():
    def __init__(self, window=3):
        self.slope = None
        self.intercept = None

    def fit(self, time, angle):
        self.slope, self.intercept, _, _, _ = linregress(time, angle)

    def predict(self, time):
        if self.slope is None:
            return None
        k = self.slope
        b = self.intercept
        return k * time + b

    def clean(self):
        self.slope = None
        self.intercept = None


class PolyPredictor():
    def __init__(self, window=3):
        self.x = None
        self.y = None

    def fit(self, time, angle):
        #self.p = np.polyfit(time, angle, deg=2)
        self.x = time
        self.y = angle

    def predict(self, time):
        if self.x is None:
            return None
        x, (x1, x2, x3), (y1, y2, y3) = time, self.x, self.y
        # Lagrange's interpolation
        return (
            y1*(x-x2)*(x-x3)/(x1-x2)*(x1-x3) +
            y2*(x-x1)*(x-x3)/(x2-x1)*(x2-x3) +
            y3*(x-x1)*(x-x2)/(x3-x1)*(x3-x2)
        )

    def clean(self):
        self.x = None
        self.y = None


class PredictProcess(multiprocessing.Process):
    def __init__(self, box_recv, pred_send):
        super(PredictProcess, self).__init__()
        self.box_recv = box_recv
        self.pred_send = pred_send
        self.if_predict = False

    def recv_box(self):
        while True:
            armor_box, begin = self.box_recv.recv()
            if self.if_predict != self.uart.predict:
                self.if_predict = self.uart.predict
                self.pred_send.send(self.if_predict)
            if armor_box is None:
                self.timestamp.clean()
                self.angles.clean()
                self.traj_predictor.clean()
                #self.uart.sendTarget(0, 0.45, 0) # no move
            else:
                if self.if_predict:
                    self.pitch = ((armor_box[1]+armor_box[3])/2 - 240) * 0.5
                else:
                    self.pitch = ((armor_box[1]+armor_box[3])/2 - 240) * 0.5
                self.distance = (32 * 400) / (armor_box[3] - armor_box[1])
                self.yaw = math.atan(
                    ((armor_box[0] + armor_box[2])/2 - 320) / 652) / math.pi * 180
                if self.if_predict:
                    if abs(self.yaw) < 3: # only move when target close
                        self.uart.sendTarget(0, self.pitch, self.distance)
                    self.timestamp.put(begin-0.01)  # reduce time to capture image
                    self.angles.put(self.yaw)
                    if self.angles.full:
                        last_angles = self.angles.getAll()
                        last_timestamps = self.timestamp.getAll()
                        self.traj_predictor.fit(last_timestamps, last_angles)
                else:
                    self.uart.sendTarget(
                        self.yaw * 0.5, self.pitch, self.distance)

    def run(self):
        self.uart = Uart()
        self.traj_predictor = LinearPredictor()
        self.angles = Memory()
        self.timestamp = Memory()
        self.distance = 0
        self.pitch = 0.45
        self.yaw = 0
        shoot_available = 2
        time_step = 0.0001
        enemy_color = self.uart.enemy_color
        while enemy_color is None:
            print("Wait for color...")
            time.sleep(1.0/30)
            enemy_color = self.uart.enemy_color
        self.pred_send.send(enemy_color)
        self.recv_thread = threading.Thread(target=self.recv_box)
        self.recv_thread.start()
        while True:
            if self.uart.predict:
                next_angle = self.traj_predictor.predict(
                    time.time()+0.4)
                if next_angle is None:
                    time.sleep(time_step)
                    continue
                if abs(next_angle) < 1.5:
                    if shoot_available > 0:
                        print("Shoot !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        self.uart.sendTarget(-0.4, 0.45, 0) # shoot without move
                        shoot_available -= 1
                else:
                    shoot_available = 2
            time.sleep(time_step)


class FSM():
    def __init__(self):
        self.state = "detecting"
        self.tracking_ctr = 0

    def transfer(self, s):
        print("Transfer from {} to {}".format(self.state, s))
        self.state = s
        self.tracking_ctr = 0

    def run(self, box):
        if self.state == "detecting":
            if box is None:
                return
            #self.transfer("tracking")
            return
        elif self.state == "tracking":
            if box is None:
                self.tracking_ctr = 0
                self.transfer("detecting")
                return
            self.tracking_ctr += 1
            if self.tracking_ctr > 60:
                self.tracking_ctr = 0
                self.transfer("detecting")
                return


class DetectProcess(multiprocessing.Process):
    def __init__(self, box_send, pred_recv):
        super(multiprocessing.Process, self).__init__()
        self.box_send = box_send
        self.pred_recv = pred_recv
        self.if_predict = False

    def recv_predict(self):
        self.if_predict = self.pred_recv.recv()
        self.fsm.transfer("detecting")

    def run(self):
        from camera import Camera
        from detect_image import RFBNetDetector
        from DaSiamRPN import DaSiamRPN
        armor_box = None
        last_armor_box = None
        camera = Camera()
        rfb_net = RFBNetDetector()
        tracker = DaSiamRPN()
        fsm = FSM()
        self.fsm = fsm
        src = camera.src
        while src is None:
            print("Wait for camera...")
            time.sleep(0.01)
            src = camera.src
        print("Wait for passing color ...")
        enemy_color = self.pred_recv.recv()
        print("Got passing color ...")
        self.recv_thread = threading.Thread(target=self.recv_predict)
        self.recv_thread.start()

        while True:
            begin = time.time()
            src = camera.src.copy()
            if self.if_predict:
                boxes = rfb_net.detect(src, thresh=0.5)  # [class, image]
                # [[x1,y1,x2,y2,score]*n]
                boxes = np.array(boxes[[1, 2][enemy_color == "red"]][0])
                if boxes.size == 0:
                    armor_box = None
                    last_armor_box = None
                else:
                    confidence = boxes[:, -1]
                    max_arg = np.argmax(confidence)
                    armor_box = boxes[max_arg, :4]
                    if boxes.size >= 2 and last_armor_box is not None:
                        confidence[max_arg] = np.min(confidence)
                        max_arg = np.argmax(confidence)
                        sec_armor_box = boxes[max_arg,:4]
                        if abs(armor_box[0]-last_armor_box[0]) > last_armor_box[2]*0.5 or abs(armor_box[1]-last_armor_box[1]) > last_armor_box[3]*0.5:
                            if abs(sec_armor_box[0]-last_armor_box[0]) < last_armor_box[2]*0.5 and abs(sec_armor_box[1]-last_armor_box[1]) < last_armor_box[3]*0.5:
                                armor_box = sec_armor_box
                    last_armor_box = armor_box
            else:
                if self.fsm.state == "detecting":
                    boxes = rfb_net.detect(src, thresh=0.5)
                    boxes = np.array(boxes[[1, 2][enemy_color == "red"]][0])
                    if boxes.size == 0:
                        armor_box = None
                    else:
                        confidence = boxes[:, -1]
                        max_arg = np.argmax(confidence)
                        armor_box = boxes[max_arg, :4]
                        #tracker.init(src, armor_box)
                        #print("Track init :{}".format(armor_box))
                elif self.fsm.state == "tracking":
                    bbox, score = tracker.update(src)
                    armor_box = bbox if score > 0.985 else None
                self.fsm.run(armor_box)

            self.box_send.send((armor_box, begin))

            if True:
                if armor_box is None:
                    cv2.imshow("src", src)
                    cv2.waitKey(1)
                else:
                    x1, y1, x2, y2 = armor_box
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    src = cv2.rectangle(src, (x1, y1), (x2, y2),
                                        (0, 255, 0), 2)
                    cv2.imshow("src", src)
                    cv2.waitKey(1)
            end = time.time()
#            print("FPS", 1/(end - begin))


if __name__ == '__main__':
    (box_recv, box_send) = multiprocessing.Pipe(duplex=False)
    (pred_recv, pred_send) = multiprocessing.Pipe(duplex=False)

    predictor = PredictProcess(box_recv, pred_send)
    predictor.start()

    detector = DetectProcess(box_send, pred_recv)
    detector.start()
