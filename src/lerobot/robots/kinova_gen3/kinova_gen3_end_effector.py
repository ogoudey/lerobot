import logging
import time
from typing import Any
import sys
import os
import threading
from functools import cached_property

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from .config_kinova_gen3_end_effector import KinovaGen3EndEffectorConfig

logger = logging.getLogger(__name__)

from ..robot import Robot

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

TIMEOUT_DURATION=20.0

class KinovaGen3EndEffector(Robot):
    """
    WIP
    """

    config_class = KinovaGen3EndEffectorConfig
    name = "kinova_gen3_end_effector"

    def __init__(self, config: KinovaGen3EndEffectorConfig):
        super().__init__(config)
        self.goal_position = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "theta_x": 0.0, "theta_y": 0.0, "theta_z": 0.0, "gripper": 0.0}
        self.config = config
    def ee_writer(self):
        print(f"Importing utilities")
        from . import utilities
        print(f"{utilities.__file__}")
        args = utilities.parseConnectionArguments()
        print(f"{args}")
        with utilities.DeviceConnection.createTcpConnection(args) as router:

        # Create required services
            self.base = BaseClient(router)
            self.base_cyclic = BaseCyclicClient(router)
            
            # Example core
            success = True
            print(f"Moving to home position")
            success &= self.move_to_home_position()
            
            try:
                while True:
                    success &= self.send_cartesian()
            except KeyboardInterrupt:
                print(f"Exiting action thread")
                self.running = False

    def connect(self, calibrate=False):
        # STart action thread
        
        """
        try:
            while True:
                if not self.running:
                    self.t.join()
        except KeyboardInterrupt:
            print(f"Exiting joiner thread")
            self.connected = False
            self.running = False
        """
        pass
    
    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        
        return self._motors_ft

    def calibrate(self):
        pass

    def configure(self):
        self.t = threading.Thread(target=self.ee_writer)
        self.t.start()
        self.running = True
        pass

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "l0": float,
            "l1": float,
            "l2": float,
            "l3": float,
            "l4": float,
            "l5": float,
            "l6": float
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:

        return {
            "front": (480, 640, 3),
            "onboard": (720, 1280, 3)
        } # change to onboard camera width?
        

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def get_observation(self):
        # Read arm position
        """
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        """
        return dict()

    @property
    def is_calibrated(self):
        return True # CHANGE?

    def disconnect(self):
        pass

    @property
    def is_connected(self) -> bool:
        return self.connected

    def send_cartesian(self):
        #print("Starting Cartesian action movement ...")
        action = Base_pb2.Action()
        action.name = "Example Cartesian action movement"
        action.application_data = ""

        feedback = self.base_cyclic.RefreshFeedback()
        #print(f"Goal: {self.goal_position}")
        #print(f"Current position: {feedback.base.tool_pose_x} {feedback.base.tool_pose_y} {feedback.base.tool_pose_z}")
        #input("[enter]")
        cartesian_pose = action.reach_pose.target_pose
        cartesian_pose.x = feedback.base.tool_pose_x + self.goal_position["delta_x"]          # (meters)
        cartesian_pose.y = feedback.base.tool_pose_y + self.goal_position["delta_y"]  
        cartesian_pose.z = feedback.base.tool_pose_z + self.goal_position["delta_z"]  
        cartesian_pose.theta_x = feedback.base.tool_pose_theta_x + self.goal_position["theta_x"]  
        cartesian_pose.theta_y = feedback.base.tool_pose_theta_y + self.goal_position["theta_y"]  
        cartesian_pose.theta_z = feedback.base.tool_pose_theta_z + self.goal_position["theta_z"]  

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        #print("Executing action")
        self.base.ExecuteAction(action)

        #print("Waiting for movement to finish ...")
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            #print("Cartesian movement completed")
            pass
        else:
            #print("Timeout on action notification wait")
            pass
        return finished

    def send_action(self, action):
        self.goal_position = action

    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e = e):
            print("EVENT : " + \
                Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def move_to_home_position(self):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        
        # Move arm to ready position
        print("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            print("Can't reach safe position. Exiting")
            return False

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self.check_for_end_or_abort(e),
            Base_pb2.NotificationOptions()
        )

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

        if finished:
            print("Safe position reached")
        else:
            print("Timeout on action notification wait")
        return finished