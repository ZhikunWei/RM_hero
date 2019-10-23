import time
import math
import threading
from collections import namedtuple

import cv2
import numpy as np
from scipy.stats import linregress

from camera import Camera
from detect_image import RFBNetDetector
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

class Predictor():
    def __init__(self, window=3):
        self.slope = None 
        self.intercept = None
    
    def fit(self, time, angle):
#        self.p = np.polyfit(time, angle, w=self.weight, deg=2)
        self.slope, self.intercept,_,_,_ = linregress(time, angle)

    def predict(self, time):
        if self.slope is None:
            return None
        k = self.slope
        b = self.intercept
        return k * time + b
        
    def clean(self):
        self.slope = None
        self.intercept = None

uart = Uart()    
predictor = Predictor()
distance = 300
pitch = 0
yaw = 0

def predict_shoot():
    global uart, predictor, distance, pitch, yaw
    shoot_available = 2
    while True:
        next_angle = predictor.predict(time.time()+0.4)
        if next_angle is None:
            time.sleep(0.001)
            continue
        
        if uart.predict:
#            print("Next angle: {}".format(next_angle))
            if abs(next_angle) < 1.5:
                if shoot_available > 0:
                    print("Shoot !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    uart.sendTarget(-0.4, pitch, distance)
                    shoot_available -= 1
            else:
                shoot_available = 2
        time.sleep(0.001)
    

t = threading.Thread(target=predict_shoot)
#t.setDaemon(True)
t.start()

def run():
    global uart, predictor, distance, pitch, yaw
    detector = RFBNetDetector()
    camera = Camera()
    angles = Memory()
    timestamp = Memory()

    enemy_color = uart.enemy_color
    while enemy_color is None:
        print("Wait for color...")
        enemy_color = uart.enemy_color
        time.sleep(0.0333)
    src = camera.src
    while src is None:
        print("Wait for camera...")
        src = camera.src
        time.sleep(0.01)
        
    armor_box = None
    last_armor_box = None
    uart_angle = None

    while True:	
        begin = time.time()
        uart_angle = (uart.angle)
        enemy_color = uart.enemy_color
        src = camera.src.copy()
            
        boxes = detector.detect(src)
        boxes = np.array(boxes[[1,2][enemy_color=="red"]][0])
        #print(boxes)
        if boxes.size == 0:
            armor_box = None
            last_armor_box = None
        else:
            confidence = boxes[:,-1]
            max_arg = np.argmax(confidence)
            armor_box = boxes[max_arg,:4]
            
            if boxes.size >= 2 and last_armor_box is not None:
                confidence[max_arg] = np.min(confidence)
                max_arg = np.argmax(confidence)
                sec_armor_box = boxes[max_arg,:4]
                if abs(armor_box[0]-last_armor_box[0]) > last_armor_box[2]*0.5 or abs(armor_box[1]-last_armor_box[1]) > last_armor_box[3]*0.5:
                    if abs(sec_armor_box[0]-last_armor_box[0]) < last_armor_box[2]*0.5 and abs(sec_armor_box[1]-last_armor_box[1]) < last_armor_box[3]*0.5:
                        armor_box = sec_armor_box
            last_armor_box = armor_box

        if armor_box is None:
            angles.clean()
            timestamp.clean()
            predictor.clean()
            cv2.imshow("src", src)
            cv2.waitKey(1)
            continue

        pitch = ((armor_box[1]+armor_box[3])/2 - 240) * 0.5
        distance = (30 * 400) / (armor_box[3] - armor_box[1])
        x_error = math.atan(((armor_box[0] + armor_box[2])/2 - (335+390)/2) / 652) / math.pi * 180
        yaw = x_error * 0.58
        timestamp.put(begin-0.01)
        angles.put(x_error)

        if angles.full:
            last_angles = angles.getAll()
            last_timestamps = timestamp.getAll()
            predictor.fit(last_timestamps, last_angles)
            print("Last angles: {}".format(last_angles))
            x = x_error * 0.58 # + omega * 1.8
        else:
            x = (x_error) * 0.58 #+ 1.6

        z = distance
        y = pitch
        if not uart.predict:
            uart.sendTarget(x, y, z)
        else:
            uart.sendTarget(0, y, z)
        end = time.time()
        #print("Box: {}, angle: {}, send: {}".format(armor_box, uart_angle, (x, y, z)))
        if True:
            x1, y1, x2, y2 = armor_box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            src = cv2.rectangle(src, (x1, y1), (x2, y2), 
                (0,255,0), 2)
            if last_armor_box is not None:
                x1, y1, x2, y2 = last_armor_box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                src = cv2.rectangle(src, (x1, y1), (x2, y2), 
                    (255,0,255), 2)

            cv2.imshow("src", src)
            cv2.waitKey(1)
        #print("FPS", 1/(end - begin))


if __name__ == '__main__':
    run()
