from camera import Camera
from uart import Uart
from detect_image import RFBNetDetector
#from SiamFC import TrackerSiamFC
import cv2
import time
from collections import namedtuple

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
            #self.transfer("detected")
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

Bbox = namedtuple('Bbox', ['x', 'y', 'width', 'height'])
camera = Camera()
uart = Uart()
detector = RFBNetDetector()
tracker = cv2.TrackerKCF_create()
fsm = FSM()


time.sleep(1)
def run():
    global camera, uart
    src = camera.src
    enemy_color = uart.enemy_color
    while enemy_color is None:
        print("Wait for color...")
        enemy_color = uart.enemy_color
        time.sleep(0.0333)
        
    armor_box = None
    x_last_err = 0

    while True:	
        begin = time.time()
        src = camera.src.copy()
        #armor_box, color = detector.detect(src)

        if fsm.state == "detecting":
            armor_box, color = detector.detect(src)
            if color != enemy_color:
                armor_box = None
        elif fsm.state == "detected":
            x1, y1, x2, y2 = armor_box
            x,y,w,h = x1, y1, x2-x1, y2-y1
#            frame = cv2.resize(src, (320, 240))
            tracker.init(src, (x,y,w,h))
            x_last_err = 0
            print("Track init :{}".format(armor_box))
        elif fsm.state == "tracking":
#            frame = cv2.resize(src, (320, 240))
            res, bbox = tracker.update(src)
            x, y, w, h = bbox
#            x,y,w,h = 2*x,2*y,2*w,2*h
            if True:
                armor_box = [x,y,x+w,y+h]
                print("Tracking x,y: {}, res:{}".format([x+w/2, y+h/2], res))
            else:
                armor_box = None
            
        fsm.run(armor_box)
        
        if armor_box is None:
            x, y, z = 0, 0, 300
            uart.sendTarget(x,y,z)
            continue
        x_error = (armor_box[0]+armor_box[2])/2 - (330+400)/2
        #if x_error > 20:
        #    x = (x_last_err*0.9 + x_error) * 1.5
        #    x = min(x, 360)
        #    x = max(x, -360)
        #    x_last_err = x_error
        #else:
        x_last_err = 0
        x = (x_error) * 0.06 #+ 1.6
        y = ((armor_box[1]+armor_box[3])/2 - 240) * 0.059
        z = 300

        uart.sendTarget(x, y, z)
        print(armor_box)
        #print(x, y, z)
        #print(src.shape())
        if src is None:
            continue
        end = time.time()
        x1, y1, x2, y2 = armor_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        src = cv2.rectangle(src, (x1, y1), (x2, y2), 
              (0,255,0), 2)
        cv2.imshow("src", src)
        cv2.waitKey(1)
        #print("FPS", 1/(end - begin))


if __name__ == '__main__':
    run()

