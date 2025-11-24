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


UNITY_AVAILABLE = True
try: 
    HOST = "127.0.0.1"
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


def pose_listener(shared):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        print("Python server listening...")

        conn, addr = s.accept()
        print("Connected:", addr)

        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                transform = json.loads(data.decode().strip())
                shared["x"] = transform["px"]
                shared["y"] = transform["py"]
                shared["z"] = transform["pz"]
                shared["rx"] = transform["rx"]
                shared["ry"] = transform["ry"]
                shared["rz"] = transform["rz"]
                shared["rw"] = transform["rw"]
                shared["gripper"] = 0.0

class UnityEndEffectorTeleop(Teleoperator):
    config_class = UnityEndEffectorTeleopConfig
    name = "unity"

    def __init__(self, config: UnityEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}
        
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

        self.target_pos = {
            "x": 0.2,
            "y": 0,
            "z": 0.2,
            "roll": 0.0,
            "pitch": 90.0,
            "gripper": 0.0,
        }
        

        self.target_pos = {"px":0.0,"py":0.0,"pz":0.0,"rx":0.0,"ry":0.0,"rz":0.0,"rw":1.0}
        
        print(f"Loading URDF from: {self.urdf_path} (is file? {os.path.isfile(self.urdf_path)})")
        self.kinematics = RobotKinematics(self.urdf_path, 'gripper_frame_link', self.joint_names)
        
        # Checking order of joints so solver is aligned #
        kinematics_joint_order = list(self.kinematics.robot.model.names)[2:]
        assert kinematics_joint_order == self.joint_names
        assert self.kinematics.joint_names == self.joint_names
        
        t = threading.Thread(target=pose_listener, args=[self.target_pos])
        t.start()


    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }


    def target_to_most_recent_pos(self):
        # This should just get the last_pos, which is updated by a stream client of Unity.
        # Unity: serves a pose
        self.target_pos = self.target_pos

    def get_action(self) -> dict[str, Any]:
        print("[DEBUG] get_action() called")

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "Unity is not connected. You need to run `connect()` before `get_action()`."
            )

        self.target_to_most_recent_pos()

       

        return self.target_pos

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        pass