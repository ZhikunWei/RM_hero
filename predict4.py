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
        self.FPS = 30
        self.x_last = None

    def put(self, x, y):
        self.x.put(x)
        self.y.put(y)
        self.x_last = x

    def fit(self):
        if self.x.full:
            x, y = self.x.getAll(), self.y.getAll()
            delta = y[1:] - y[:-1]
            mu, std = delta.mean(), delta.std()
            nonzero_id = np.nonzero(abs(delta-mu)>1.5*std)[0]
            if len(nonzero_id) >= 2:
                self.T = int((nonzero_id[1:] - nonzero_id[:-1]).mean() * 2)
                if self.T > 30:
                    self.T = None
                self.FPS = 1.0/((x[-30:] - x[-31:-1]).mean())
                print("Period: {}".format(self.T))

    def predict(self, x):
        if self.T is None:
            return None
        y = self.y.getAll()
        i = int((x - self.x_last) / (1/self.FPS))
        if self.T is None:
            return None
        else:
            i %= self.T
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
            #print("FPS: {}".format(1/(time_now - self.last_recv_time)))
            self.last_recv_time = time_now
            if self.if_predict != self.uart.predict:
                self.if_predict = self.uart.predict
                self.pred_send.send(self.if_predict)
                print("Change predict to: {}".format(self.if_predict))
            if armor_box is None:
                self.traj_predictor.clean()
                self.champion_predictor.clean()
                # self.uart.sendTarget(0, 0.45, 0) # no move
            else:
                self.pitch = ((armor_box[1]+armor_box[3])/2 - 255) * 0.8
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
        self.champion_predictor = FFTPredictor(window=45)
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
                    shoot_available = 4
                    time.sleep(INTERVAL_LONG)
                    continue
                if abs(next_angle) < 1.5:
                    if shoot_available > 0:
                        #print("Shoot !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        # shoot without move
                        self.uart.sendTarget(-0.4, 0.45, 0)
                        shoot_available -= 1
                    else:
                        print("No available")
                else:
                    next_angle = self.champion_predictor.predict(time.time()+0.2+self.target_distance/1400)
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

def getDistance(armor_box):
    dis = 300
    gray = cv2.cvtColor(armor_box, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h = 0
    for con in contours:
        box = cv2.boundingRect(con)
        h = max(h, box[3])
    if h > 53:
        dis = 100
    elif h >= 36:
        dis = 150 - (h-36)/17*50
    elif h >= 27:
        dis = 200 - (h-27)/9*50
    elif h >= 22:
        dis = 250 - (h-22)/5*50
    elif h >= 18:
        dis = 300 - (h-18)/4*50
    elif h >= 16:
        dis = 350 - (h-16)/2 * 50
    elif h >= 14:
        dis = 400 - (h-14)/2 *50
    elif h >= 13:
        dis = 450
    elif h >= 12:
        dis = 500
    elif h >= 11:
        dis = 550
    elif h >= 10:
        dis = 600
    elif h >= 9:
        dis = 650
    else:
        dis = 700
    # threshold 100, camera 10ms, gain 30
    # cm - pixal
    # 100- 53
    # 150-36
    # 200-27
    # 250-22
    # 300-18
    # 350-16
    # 400 -14
    # 450 -13
    # 500-12
    # 550 -11
    # 600-10
    # 650-700 - 9
    # 740 - 8
    
    
    # threshold 200, camera 10ms, gain 30
    # 100cm - 46 pixal
    # 150cm - 31 pixal
    # 200cm - 23 pix
    # 250 cm - 18 pixal
    # 300 cm - 16 pixal
    # 350 cm - 13 pixal
    # 400 cm - 11 pixal 
    # 450 cm - 10 pixal
    # 470 cm - 9 pixal
    # 490 cm - 8 
    # 520 cm - 7
    # 600 cm - 6
    # 680 cm - 5
    # 740 cm - 4-
    return dis

class DetectProcess(multiprocessing.Process):
    def __init__(self, box_send, pred_recv):
        super(multiprocessing.Process, self).__init__()
        self.box_send = box_send
        self.pred_recv = pred_recv
        self.if_predict = False

    def recv_predict(self):
        self.if_predict = self.pred_recv.recv()
        #self.fsm.transfer("detecting")

    def run(self):
        from camera import Camera
        from detect_image import RFBNetDetector
        from DaSiamRPN import DaSiamRPN
        armor_box = None
        camera = Camera()
        rfb_net = RFBNetDetector()
        tracker = DaSiamRPN()
        fsm = FSM()
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
        dis = 200
        while True:
            fps_begin = time.time()
            begin, src = camera.timestamp, camera.src.copy()
            if self.if_predict:
                boxes = rfb_net.detect(src, thresh=0.3)  # [class, image]
                # [[x1,y1,x2,y2,score]*n]
                boxes = np.array(boxes[[1, 2][enemy_color == "red"]][0])
                if boxes.size == 0:
                    armor_box = None
                else:
                    confidence = boxes[:, -1]
                    max_arg = np.argmax(confidence)
                    armor_box = boxes[max_arg, :4]
            else:
                if fsm.state == "detecting":
                    boxes = rfb_net.detect(src, thresh=0.3)
                    boxes = np.array(boxes[[1, 2][enemy_color == "red"]][0])
                    if boxes.size == 0:
                        armor_box = None
                    else:
                        confidence = boxes[:, -1]
                        max_arg = np.argmax(confidence)
                        armor_box = boxes[max_arg, :4]
                        #tracker.init(src, armor_box)
                        #print("Track init :{}".format(armor_box))
                elif fsm.state == "tracking":
                    bbox, score = tracker.update(src)
                    armor_box = bbox if score > 0.98 else None
                fsm.run(armor_box)
            if armor_box is not None:
                x1, y1, x2, y2 = armor_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #print(src.shape, armor_box)
                if x1<x2-2 and y1<y2-2 and 0<x1<640 and 0<x2<640 and 0<y1<480 and 0<y2<480:
                    cur_dis = getDistance(src[y1:y2, x1:x2])
                    dis = cur_dis * 0.5 + dis *0.5
                else:
                    dis = 200
                #armor_box = armor_box.tolist()
                #armor_box.append(dis)
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
                    src = cv2.putText(src, "%.2f"%dis, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)
                    cv2.imshow("src", src)
                    cv2.waitKey(1)
            #print("FPS: ", 1/(time.time() - fps_begin))


if __name__ == '__main__':
    (box_recv, box_send) = multiprocessing.Pipe(duplex=False)
    (pred_recv, pred_send) = multiprocessing.Pipe(duplex=False)

    predictor = PredictProcess(box_recv, pred_send)
    predictor.start()

    detector = DetectProcess(box_send, pred_recv)
    detector.start()
