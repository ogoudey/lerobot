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


UNITY_AVAILABLE = False
try:
    pass
    # import line
except ImportError:
    keyboard = None
    UNITY_AVAILABLE = False
except Exception as e:
    keyboard = None
    UNITY_AVAILABLE = False
    logging.info(f"Could not import Unity stuff: {e}")


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
        
        print(f"Loading URDF from: {self.urdf_path} (is file? {os.path.isfile(self.urdf_path)})")
        self.kinematics = RobotKinematics(self.urdf_path, 'gripper_frame_link', self.joint_names)
        
        # Checking order of joints so solver is aligned #
        kinematics_joint_order = list(self.kinematics.robot.model.names)[2:]
        assert kinematics_joint_order == self.joint_names
        assert self.kinematics.joint_names == self.joint_names
     
    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return UNITY_AVAILABLE

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                "Unity is already connected. Do not run `robot.connect()` twice."
            )

        if UNITY_AVAILABLE:
            logging.info("Unity is available - doing something?.")
        else:
            logging.info("Unity not available - skipping local keyboard listener.")

    def calibrate(self) -> None:
        pass

    def configure(self):
        pass

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