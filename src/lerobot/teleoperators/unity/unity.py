import logging
import os
import sys
import time
from queue import Queue
from typing import Any

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_unity import UnityEndEffectorTeleopConfig

from ...model.kinematics import RobotKinematics    

import cv2
import socket as s
import time
import base64
import random
    
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

UNITY_AVAILABLE = True
try: 
    HOST = "0.0.0.0"
    PORT = 5001

    import socket
    import threading
    import json
except ImportError:
    UNITY_AVAILABLE = False
    raise ImportError(f"Could not import Unity stuff: {e}")
except Exception as e:
    UNITY_AVAILABLE = False
    logging.info(f"Could not import Unity stuff: {e}")
    raise Exception(f"Could not import Unity stuff: {e}")

def termux_listener(shared):
    UDP_PORT = 5005

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", UDP_PORT))

    print(f"Listening on UDP port {UDP_PORT}...")

    prev_time = None

    vel_x = vel_y = vel_z = 0.0
    pos_x = pos_y = pos_z = 0.0

    theta_x = theta_y = theta_z = 0.0
    # (overriding target_pos)

    while True:
        print(f"Waiting for first data")
        data, addr = sock.recvfrom(4096)
        msg = json.loads(data.decode())
        
        try:
            print("Received:", msg)
            now = time.time()
            if prev_time is None:
                prev_time = now
                continue
            dt = now - prev_time
            prev_time = now

            # Delta
            accel_x = msg["ax"]
            accel_y = msg["ay"]
            accel_z = msg["az"]

            # output deltas
            delta_x = accel_x * dt*dt /2
            delta_y = accel_y * dt*dt /2
            delta_z = accel_z * dt*dt /2

            # Theta
            gyro_x = msg["gx"]
            gyro_y = msg["gy"]
            gyro_z = msg["gz"]


            theta_x = gyro_x * dt
            theta_y = gyro_y * dt
            theta_z = gyro_z * dt

            shared["delta_x"] = delta_x
            shared["delta_y"] = delta_y
            shared["delta_z"] = delta_z
            shared["theta_x"] = theta_x
            shared["theta_y"] = theta_y
            shared["theta_z"] = theta_z
            shared["gripper"] = 0.0
            print("Calculated:", shared)
        except KeyError:
            print(f"{msg} is not the expected transform format...")

# Unity pose listener. Now with gripper too.
def pose_listener(shared):
    heard_poses = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        print("Pose server listening for pose...")

        conn, addr = s.accept()
        print("Connected:", addr)
        # 'gripper': 0.0, 'delta_x': 0.06789376088414159, 'delta_y': -0.060046243950372995, 'delta_z': -0.02887945867382694, 'theta_x': 0.0007037802541721705, 'theta_y': -0.0007037802541721705, 'theta_z': 0.0}
        last_pose = {}
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                try:
                    transform_gripper = json.loads(data.decode().strip())
                except Exception as e:
                    print(f"BAd data: {e}. (Continuing)")
                    continue
                #print(f"Updating local data: {transform}")
                #print(f"{heard_poses} poses heard; input: {transform_gripper}")

            
                try:
                    if "px" in last_pose:
                        shared["delta_x"] = transform_gripper["px"] - last_pose["px"]
                        shared["delta_y"] = transform_gripper["py"] - last_pose["py"]
                        shared["delta_z"] = transform_gripper["pz"] - last_pose["pz"]

                        q_curr = [transform_gripper["rx"], transform_gripper["ry"], transform_gripper["rz"], transform_gripper["rw"]]
                        q_last = [last_pose["rx"], last_pose["ry"], last_pose["rz"], last_pose["rw"]]
                        r_delta = R.from_quat(q_curr) * R.from_quat(q_last).inv()

                        roll, pitch, yaw = r_delta.as_euler("xyz", degrees=False)
                        shared["theta_x"], shared["theta_y"], shared["theta_z"] = float(roll), float(pitch), float(yaw)
                        shared["gripper"] = transform_gripper["gripper"]

                        
                    last_pose["px"] = transform_gripper["px"]
                    last_pose["py"] = transform_gripper["py"]
                    last_pose["pz"] = transform_gripper["pz"]
                    
                    last_pose["rx"] = transform_gripper["rx"]
                    last_pose["ry"] = transform_gripper["ry"]
                    last_pose["rz"] = transform_gripper["rz"]
                    last_pose["rw"] = transform_gripper["rw"]
                    if shared["theta_y"] > 0.01:
                        print(f"[pose_listener] {shared}")
                except KeyError as e:
                    print(f"{transform_gripper} is not the expected transform format... ({e})")
                except Exception as e:
                    print(f"Error: {e}")
                heard_poses += 1          

class UnityEndEffectorTeleop(Teleoperator):
    config_class = UnityEndEffectorTeleopConfig
    name = "unity"

    def __init__(self, config: UnityEndEffectorTeleopConfig, with_ik=False):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.fps = 30
        self.display_data = False
        self.teleop_time_s = 400
        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}

        if hasattr(config, "unity_projector"):
            self.unity_projector = config.unity_projector
        
        self.urdf_path = os.path.abspath("custom_brains/so101_new_calib.urdf")
        #self.urdf_path = os.path.abspath("custom_brains/so101_old_calib.urdf")
        
        max_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ] # for reference
        
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self.target_pos = {}
        """ # For SO101
        self.target_pos = {
            "x": 0.2,
            "y": 0,
            "z": 0.2,
            "roll": 0.0,
            "pitch": 90.0,
            "gripper": 0.0,
        }
        """
        if with_ik:
            print(f"Loading URDF from: {self.urdf_path} (is file? {os.path.isfile(self.urdf_path)})")
            self.kinematics = RobotKinematics(self.urdf_path, 'gripper_frame_link', self.joint_names)
            
            # Checking order of joints so solver is aligned #
            kinematics_joint_order = list(self.kinematics.robot.model.names)[2:]
            assert kinematics_joint_order == self.joint_names
            assert self.kinematics.joint_names == self.joint_names
        
        #self.t = threading.Thread(target=pose_listener, args=[self.target_pos])
        self.transform = threading.Thread(target=pose_listener, args=[self.target_pos])
        
        self.socket = s.socket(socket.AF_INET, socket.SOCK_STREAM)        

    def project(self, raw_frame):
        # --- Convert frame to PNG bytes ---
        try:
            ret2, buffer = cv2.imencode('.png', raw_frame)
            if not ret2:
                return

            # --- Base64 encode ---
            b64_data = base64.b64encode(buffer).decode('utf-8')

            data_bytes = b64_data.encode('utf-8')
            length = len(data_bytes)
            self.socket.sendall(length.to_bytes(4, 'big') + data_bytes)
            return True
        except Exception as e:
            print(f"{e}\n Failed to send frame to Unity.")
            return False

    def calibrate(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def feedback_features(self) -> dict:
        return {}

    def configure(self):
        pass

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(list(self.target_pos.keys()))),
            "names": {},
        }

    def connect(self, calibrate=False) -> None:
        # Open socket to Unity
        print("Connecting...")
        self.transform.start()
        self.connected = True
        print(f"Waiting for teleop data... (delta_x not in {self.target_pos})", end="\n")
        while not "delta_x" in self.target_pos: # until the x target moves from its initial pose (the teleop data is doing something...)
            if random.random() < 0.1:
                print(f"Waiting for teleop data... (delta_x not in {self.target_pos})", end="\n")
            time.sleep(0.1)
        print(f"Connected to teleop data")
        try:
            self.socket.connect(("192.168.0.209", 5000)) # VR computer 
            print(f"Successfully connected to Unity VR")
        except Exception as e:
            print(f"Streaming Connection Error: {e}")
        if UNITY_AVAILABLE:
            logging.info("Unity is available!")

    @property
    def is_connected(self) -> bool:
        return self.connected

    def get_action(self) -> dict[str, Any]:
        
        if random.random() < 0.005:
            print(f"Target pos: {self.target_pos}")

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Unity is not connected. You need to run `connect()` before `get_action()`."
            )

        return self.target_pos

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        pass

if __name__ == "__main__":
    cfg = UnityEndEffectorTeleopConfig()
    ut = UnityEndEffectorTeleop(cfg)
    

