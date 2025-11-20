import cv2
from threading import Thread
import time

class CameraReader(Thread):
    """ Class that provides a .frame and updates it as a parallel thread. """
    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.running = True
        
        self.frame_updates = 0

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                #self.frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 240)).copy()
                #self.frame = cv2.resize(frame, (320, 240)).copy()
                #self.frame = cv2.resize(frame, (512, 512)).copy()
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
                self.frame_updates += 1
                #print("Grab?", self.cap.grab())
                #print("\rUpdated frame x", self.frame_updates, end="\n")
            else:
                print("\rNo retrieved frame yet...")
            time.sleep(0.001)  # small sleep to yield CPU

    def stop(self):
        self.running = False
