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
        self.p = None

    def fit(self, time, angle):
        self.p = np.polyfit(time, angle, deg=2)

    def predict(self, time):
        if self.p is None:
            return None
        p = self.p
        return p[0] * time**2 + p[1] * time + p[2]

    def clean(self):
        self.p = None


class PredictProcess(multiprocessing.Process):
    def __init__(self, pipe_box, pipe_predict):
        super(PredictProcess, self).__init__()
        self.pipe_box = pipe_box
        self.pipe_predict = pipe_predict
        self.trace_predictor = LinearPredictor()
        self.angles = Memory()
        self.timestamp = Memory()
        self.distance = 300
        self.pitch = 0.45
        self.yaw = 0
        self.if_predict = False

    def recv_box(self):
        while True:
            armor_box, begin = self.pipe_box.recv()
            if_predict = self.uart.predict
            #print("Send: {}".format(if_predict))
            self.pipe_predict.send(if_predict)
            if armor_box is None:
                self.timestamp.clean()
                self.angles.clean()
                self.trace_predictor.clean()
                self.uart.sendTarget(0, 0.35, 0)
            else:
                self.pitch = ((armor_box[1]+armor_box[3])/2 - 235) * 0.01
                self.distance = (30 * 400) / (armor_box[3] - armor_box[1])
                self.yaw = math.atan(
                    ((armor_box[0] + armor_box[2])/2 - (335+390)/2) / 652) / math.pi * 180
                if if_predict:
                    self.uart.sendTarget(0, self.pitch, self.distance)
                else:
                    self.uart.sendTarget(self.yaw * 0.55, self.pitch, self.distance)
                self.timestamp.put(begin-0.01)  # reduce time to capture image
                self.angles.put(self.yaw)
            if if_predict and self.angles.full:
                last_angles = self.angles.getAll()
                last_timestamps = self.timestamp.getAll()
                self.trace_predictor.fit(last_timestamps, last_angles)

    def run(self):
        self.uart = Uart()
        enemy_color = self.uart.enemy_color
        while enemy_color is None:
            print("Wait for color...")
            time.sleep(1.0/30)
            enemy_color = self.uart.enemy_color
        self.pipe_predict.send(enemy_color)
        self.recv_thread = threading.Thread(target=self.recv_box)
        self.recv_thread.start()
        shoot_available = 2
        time_step = 0.001
        while True:
            if_predict = self.uart.predict
            if if_predict:
                next_angle = self.trace_predictor.predict(time.time()+0.35)
                if next_angle is None:
                    time.sleep(time_step)
                    continue
                if abs(next_angle) < 1.0:
                    if shoot_available > 0:
                        print("Shoot !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        self.uart.sendTarget(-0.4, self.pitch, self.distance)
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
            self.transfer("detected")
            return
        elif self.state == "detected":
            self.tracking_ctr = 0
            self.transfer("tracking")
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
    def __init__(self, pipe_box, pipe_predict):
        super(multiprocessing.Process, self).__init__()
        self.pipe_box = pipe_box
        self.pipe_predict = pipe_predict
        self.fsm = FSM()
        print("Wait for passing color ...")
        self.enemy_color = self.pipe_predict.recv()
        print("Got passing color ...")
        self.if_predict = False

    def recv_predict(self):
        while True:
            uart_predict = self.pipe_predict.recv()
            if self.if_predict == True and uart_predict == False:
                self.fsm.transfer("detecting")
            self.if_predict = uart_predict
        #    print(self.if_predict)

    def run(self):
        from DaSiamRPN import DaSiamRPN
        from detect_image import RFBNetDetector
        from camera import Camera
        self.rfb_net = RFBNetDetector()
        self.tracker = DaSiamRPN()
        self.camera = Camera()
        src = self.camera.src
        while src is None:
            print("Wait for camera...")
            time.sleep(0.01)
            src = self.camera.src
        self.recv_thread = threading.Thread(target=self.recv_predict)
        self.recv_thread.start()
        armor_box = None
        while True:
            begin = time.time()
            src = self.camera.src.copy()
#            print(self.if_predict)
            if self.if_predict:
                boxes = self.rfb_net.detect(src)  # [class, image]
                # [[x1,y1,x2,y2,score]*n]
                boxes = np.array(boxes[[1, 2][self.enemy_color == "red"]][0])

                if boxes.size == 0:
                    armor_box = None
                else:
                    confidence = boxes[:, -1]
                    max_arg = np.argmax(confidence)
                    armor_box = boxes[max_arg, :4]
            else:
                if self.fsm.state == "detecting":
                    boxes = self.rfb_net.detect(src)  # [class, image]
                    # [[x1,y1,x2,y2,score]*n]
                    boxes = np.array(boxes[[1, 2][self.enemy_color == "red"]][0])
                    print(boxes)
                    if boxes.size == 0:
                        armor_box = None
                    else:
                        confidence = boxes[:, -1]
                        max_arg = np.argmax(confidence)
                        armor_box = boxes[max_arg, :4]
                    # transfer to detected instantly if armor box exists
                    #self.fsm.run(armor_box)
                elif self.fsm.state == "detected":
                    self.tracker.init(src, armor_box)
                    print("Track init :{}".format(armor_box))
                elif self.fsm.state == "tracking":
                    bbox, score = self.tracker.update(src)
                    if score > 0.99:
                        armor_box = bbox
                    else:
                        armor_box = None
                self.fsm.run(armor_box)

            self.pipe_box.send((armor_box, begin))

            if True:
                if armor_box is None:
                    cv2.imshow("src", src)
                    cv2.waitKey(1)
                    continue
                x1, y1, x2, y2 = armor_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                src = cv2.rectangle(src, (x1, y1), (x2, y2),
                                    (0, 255, 0), 2)
                cv2.imshow("src", src)
                cv2.waitKey(1)
            #print("FPS", 1/(end - begin))


if __name__ == '__main__':
    pipe_box = multiprocessing.Pipe(duplex=True)
    pipe_predict = multiprocessing.Pipe(duplex=True)

    predictor = PredictProcess(pipe_box[0], pipe_predict[0])
    predictor.start()

    detector = DetectProcess(pipe_box[1], pipe_predict[1])
    detector.start()

    #predictor.join()
    #detector.join()
