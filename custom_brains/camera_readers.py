import cv2
from threading import Thread
import time
import os
import sys

class WebcamReader(Thread):
    """ Class that provides a .frame and updates it as a parallel thread. """
    @staticmethod
    def get_cap(rstp_url):
        # OPEN AND REOPEN TRICK
        cap = cv2.VideoCapture(rstp_url)
        time.sleep(0.2)
        return cap

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

def open_cam(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Opening cam...")
    time.sleep(0.2)  # allow to apply
    return cap


import ctypes
import contextlib

# Use C-level freopen to redirect stderr temporarily
@contextlib.contextmanager
def suppress_libjpeg_warnings():
    """Suppress libjpeg 'Corrupt JPEG data' warnings in threads safely."""
    libc = ctypes.CDLL(None)
    # Save original stderr
    original_stderr = libc.stderr
    # Open /dev/null
    devnull = open("/dev/null", "w")
    # Replace stderr
    libc.stderr = ctypes.c_void_p(devnull.fileno())
    try:
        yield
    finally:
        # Restore original stderr
        libc.stderr = original_stderr
        devnull.close()


#
# v4l2-ctl --list-devices
#
# v4l2-ctl -d /dev/video{X} --list-formats
#
class USBCameraReader(Thread):
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
        self.raw_frame = None
        self.running = True
        
        self.frame_updates = 0

    def run(self):
        while self.running:

            with suppress_libjpeg_warnings():
                ret, self.raw_frame = self.cap.read()

            if ret:
                self.frame = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB).copy()
                self.frame_updates += 1
            else:
                print("\rNo retrieved frame yet...")
            
        
    def stop(self):
        self.running = False


import matplotlib.pyplot as plt



import base64
import socket

def main():

    import socket
    import time
    import base64

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Connecting...")
    #s.connect(("192.168.0.10", 5000)) # desktop on Space Popcorn
    s.connect(("192.168.0.209", 5000)) # VR computer on Ethernet
    #s.connect(("127.0.0.1", 5000))
    print("Connected!")

    #cap = WebcamReader.get_cap("rtsp://192.168.0.159:8080/h264_ulaw.sdp") # phone at home
    #cap = WebcamReader.get_cap("rtsp://10.243.122.252:8080/h264_ulaw.sdp") # phone on Tufts Secure
    cap = USBCameraReader.get_cap(0)
    print(f"Got cap: {cap}")

    sent_frames = 1
    while True:
        ret, frame = cap.read()
        if ret:
            print(f"Frame # {sent_frames}", end="\r")
            # --- Convert frame to PNG bytes ---
            ret2, buffer = cv2.imencode('.png', frame)
            if not ret2:
                continue

            # --- Base64 encode ---
            b64_data = base64.b64encode(buffer).decode('utf-8')

            data_bytes = b64_data.encode('utf-8')
            length = len(data_bytes)
            s.sendall(length.to_bytes(4, 'big') + data_bytes)

            sent_frames += 1
        else:
            print(f"No ret")
    

    #cap = USBCameraReader.get_cap(6)
def just_show():
    cap = USBCameraReader.get_cap(0)
    #cap = WebcamReader.get_cap("rtsp://10.243.122.252:8080/h264_ulaw.sdp")
    #cap = WebcamReader.get_cap("rtsp://admin:admin@192.168.1.10/color")
    print(f"Got cap: {cap}")
    time.sleep(1)
    ret = None    
    while not ret:
        ret, frame = cap.read()
        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.show()
        

    
    """
    
    
    """

if __name__ == "__main__":
    just_show()
    main()
