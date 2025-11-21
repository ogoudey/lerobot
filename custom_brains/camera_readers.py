import cv2
from threading import Thread
import time
import os
import sys

class IPWebcamReader(Thread):
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

class suppress_stderr:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_WRONLY)
        self.stderr_fd = os.dup(2)
        os.dup2(self.null_fd, 2)  # redirect stderr → /dev/null

    def __exit__(self, *args):
        os.dup2(self.stderr_fd, 2)  # restore stderr
        os.close(self.null_fd)
        os.close(self.stderr_fd)

def open_cam(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(0.2)  # allow to apply
    return cap

class LogitechReader(Thread):
    @staticmethod
    def get_cap(idx):
        # OPEN AND REOPEN TRICK
        cap = open_cam(idx)
        cap.release()
        time.sleep(0.2)
        cap = open_cam(idx)
        return cap

    def __init__(self, cap):
        super().__init__()
        self.cap = cap
        if not cap.isOpened():
            print("Camera failed to open")
            exit()

        self.frame = None
        self.running = True
        
        self.frame_updates = 0

    def run(self):
        while self.running:
            with suppress_stderr():
                ret, frame = self.cap.read()

            if ret:
                self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
                self.frame_updates += 1
            else:
                print("\rNo retrieved frame yet...")
    
    def stop(self):
        self.running = False