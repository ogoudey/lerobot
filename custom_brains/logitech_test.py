import cv2
import matplotlib.pyplot as plt

# Open cameras

import time

import os
import sys

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

# OPEN AND REOPEN TRICK
cap0 = open_cam(2)
cap0.release()
time.sleep(0.2)
cap0 = open_cam(2)

cap1 = open_cam(4)
cap1.release()
time.sleep(0.2)
cap1 = open_cam(4)

#cap0 = cv2.VideoCapture(4)   # first webcam

#cap1 = cv2.VideoCapture(2)   # second webcam

if not cap0.isOpened():
    print("Camera 0 failed to open")
    exit()

if not cap1.isOpened():
    print("Camera 1 failed to open")
    exit()

# Matplotlib setup
plt.ion()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

im0 = ax0.imshow([[0]])  # placeholder
im1 = ax1.imshow([[0]])

ax0.set_title("Camera 0")
ax1.set_title("Camera 1")

ax0.axis("off")
ax1.axis("off")

while True:
    with suppress_stderr():
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        break

    # Convert BGR → RGB
    frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    # Update images
    im0.set_data(frame0_rgb)
    im1.set_data(frame1_rgb)

    plt.pause(0.001)

cap0.release()
cap1.release()
plt.close(fig)

"""
import cv2
import matplotlib.pyplot as plt

# Open cameras
cap2 = cv2.VideoCapture(2)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap2.set(cv2.CAP_PROP_FPS, 30)

cap4 = cv2.VideoCapture(4)
cap4.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap4.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap4.set(cv2.CAP_PROP_FPS, 30)


if not cap2.isOpened():
    print("Camera 2 failed to open")
    exit()

if not cap4.isOpened():
    print("Camera 4 failed to open")
    exit()

# Matplotlib setup
plt.ion()
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

im0 = ax0.imshow([[0]])  # placeholder
im1 = ax1.imshow([[0]])

ax0.set_title("Camera 0")
ax1.set_title("Camera 1")

ax0.axis("off")
ax1.axis("off")

while True:
    ret0, frame0 = cap2.read()
    ret1, frame1 = cap4.read()

    if not ret0 or not ret1:
        break

    # Convert BGR → RGB
    frame0_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    # Update images
    im0.set_data(frame0_rgb)
    im1.set_data(frame1_rgb)

    plt.pause(0.001)

cap2.release()
cap4.release()
plt.close(fig)
"""

