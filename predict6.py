import time
import math
import threading
import multiprocessing
from collections import namedtuple

import cv2
import numpy as np
from scipy.stats import linregress
from scipy.fftpack import fft, fftfreq

from uart import Uart


class Memory():
    def __init__(self, max_size=3):
        self.max_size = max_size
        self.pt = 0
        self.size = 0
        self.memory = np.zeros(self.max_size)
        self.full = False


    def put(self, x):
        self.memory[self.pt] = x
        self.pt += 1
        if self.pt >= self.max_size:
            self.pt = 0
            self.full = True
        else:
            self.size += 1

    def getAll(self):
        zero_to_now = self.memory[:self.pt]
        older = self.memory[self.pt:]
        return np.concatenate([older, zero_to_now], axis=0)

    def clean(self):
        self.pt = 0
        self.size = 0
        self.full = False
        self.memory[:] = 0


class LinearPredictor():
    def __init__(self, window=3):
        self.x = Memory(max_size=window)
        self.y = Memory(max_size=window)
        self.slope = None
        self.intercept = None

    def put(self, x, y):
        self.x.put(x)
        self.y.put(y)

    def fit(self):
        if self.x.full:
            x, y = self.x.getAll(), self.y.getAll()
            self.slope, self.intercept, _, _, _ = linregress(x, y)

    def predict(self, x):
        if self.slope is None:
            return None
        k = self.slope
        b = self.intercept
        return k * x + b

    def clean(self):
        self.slope = None
        self.intercept = None
        self.x.clean()
        self.y.clean()


class FFTPredictor():
    def __init__(self, window=120):
        self.x = Memory(max_size=window)
        self.y = Memory(max_size=window)
        self.T = None
        self.FPS = None
        self.x_last = None

    def put(self, x, y):
        self.x.put(x)
        self.y.put(y)
        self.x_last = x

    def fit(self):
        if self.x.size > 30:
            x, y = self.x.getAll(), self.y.getAll()
            freq = abs(fft(y))[:len(y)//2]
            freq /= freq.sum()
            argmax_freq = np.argmax(freq)
            if freq[argmax_freq] > 0.2:
                FPS = 1.0/((x[-30:] - x[-31:-1]).mean())
                self.T = int(fftfreq(len(y), 1/FPS)[argmax_freq] * FPS)
                self.FPS = FPS

    def predict(self, x):
        if self.T is None:
            return None
        y = self.y.getAll()
        i = int((x - self.x_last) / (1/self.FPS)) % self.T
        return y[-1+i-self.T]

    def clean(self):
        self.T = None
        self.FPS = None
        self.x_last = None
        self.x.clean()
        self.y.clean()


class PredictProcess(multiprocessing.Process):
    def __init__(self, box_recv, pred_send):
        super(PredictProcess, self).__init__()
        self.box_recv = box_recv
        self.pred_send = pred_send
        self.if_predict = False
        self.last_recv_time = time.time()
        self.target_distance = 300

    def recv_box(self):
        while True:
            armor_box, begin = self.box_recv.recv()
            time_now = time.time()
            print("FPS: {}".format(1/(time_now - self.last_recv_time)))
            self.last_recv_time = time_now
            if self.if_predict != self.uart.predict:
                self.if_predict = self.uart.predict
                self.pred_send.send(self.if_predict)
            if armor_box is None:
                self.traj_predictor.clean()
                self.champion_predictor.clean()
                # self.uart.sendTarget(0, 0.45, 0) # no move
            else:
                self.pitch = ((armor_box[1]+armor_box[3])/2 - 240) * 0.8
                self.distance = (32 * 400) / (armor_box[3] - armor_box[1])
                self.yaw = math.atan(
                    ((armor_box[0] + armor_box[2])/2 - 320) / 652
                ) / math.pi * 180
                self.traj_predictor.put(begin, self.yaw)
                self.champion_predictor.put(begin, self.yaw)
                self.traj_predictor.fit()
                self.champion_predictor.fit()
                if self.if_predict:
                    if abs(self.yaw) < 5:  # only move when target close
                        self.target_distance = self.distance
                        self.uart.sendTarget(0, self.pitch, self.distance)
                else:
                    self.uart.sendTarget(
                        self.yaw * 0.5, self.pitch, self.distance)

    def run(self):
        self.uart = Uart()
        self.traj_predictor = LinearPredictor(window=3)
        self.champion_predictor = FFTPredictor(window=120)
        self.distance = 0
        self.pitch = 0.45
        self.yaw = 0
        shoot_available = 2
        INTERVAL_LONG = 0.01
        INTERVAL_SHORT = 0.001
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
                next_angle = self.traj_predictor.predict(time.time()+0.4)
                if next_angle is None:
                    shoot_available = 2
                    time.sleep(INTERVAL_LONG)
                    continue
                if abs(next_angle) < 1.5:
                    if shoot_available > 0:
                        print("Shoot !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        # shoot without move
                        self.uart.sendTarget(-0.4, 0.45, 0)
                        shoot_available -= 1
                else:
                    next_angle = self.champion_predictor.predict(time.time()+0.2)
                    if next_angle is None:
                        shoot_available = 2
                        time.sleep(INTERVAL_LONG)
                        continue
                    if abs(next_angle) < 1.5:
                        if shoot_available > 0:
                            print("Shoot !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                            # shoot without move
                            self.uart.sendTarget(-0.4, 0.45, 0)
                            shoot_available -= 1
                    else:
                        shoot_available = 2
                time.sleep(INTERVAL_SHORT)
            else:
                #yaw_now = self.traj_predictor.predict(time.time())
                #if yaw_now is None:
                    #time.sleep(INTERVAL_LONG)
                    #continue
                #self.uart.sendTarget(yaw_now * 0.5, self.pitch, self.distance)
                time.sleep(INTERVAL_LONG)


class DetectProcess(multiprocessing.Process):
    def __init__(self, box_send, pred_recv):
        super(multiprocessing.Process, self).__init__()
        self.box_send = box_send
        self.pred_recv = pred_recv
        self.if_predict = False

    def recv_predict(self):
        self.if_predict = self.pred_recv.recv()

    def run(self):
        from camera import Camera
        from detect_image import RFBNetDetector
        armor_box = None
        camera = Camera()
        rfb_net = RFBNetDetector()
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
            begin, src = self.camera.timestamp_next, self.camera.src_next.copy()
            boxes = rfb_net.detect(src, thresh=0.5)  # [class, image]
            # [[x1,y1,x2,y2,score]*n]
            boxes = np.array(boxes[[1, 2][self.enemy_color == "red"]][0])
            if boxes.size == 0:
                armor_box = None
            else:
                confidence = boxes[:, -1]
                max_arg = np.argmax(confidence)
                armor_box = boxes[max_arg, :4]

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


if __name__ == '__main__':
    (box_recv, box_send) = multiprocessing.Pipe(duplex=False)
    (pred_recv, pred_send) = multiprocessing.Pipe(duplex=False)

    predictor = PredictProcess(box_recv, pred_send)
    predictor.start()

    detector = DetectProcess(box_send, pred_recv)
    detector.start()
