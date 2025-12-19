import logging
import time
from typing import Any
import sys
import os
import threading
from functools import cached_property
import random
import numpy as np
from datetime import datetime
from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.robot_utils import busy_wait
import math
from .config_kinova_gen3_end_effector import KinovaGen3EndEffectorConfig

logger = logging.getLogger(__name__)

from ..robot import Robot

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

home_position = {'x': 0.5766636729240417, 'y': 0.0013102991, 'z': 0.4336315989494324, 'theta_x': 90.01219940185547, 'theta_y': 2.240478352177888e-05, 'theta_z': 89.99665069580078}

# Unclassy log function
def log(message: str, log_name: str="KinovaGen3EndEffector"):
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Define log file path
    log_path = os.path.join("logs", f"{log_name}.log")
    
    # Timestamp the message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}\n"
    
    # Append the message to the log file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(formatted_message)


TIMEOUT_DURATION=20.0

class KinovaGen3EndEffector(Robot):
    """
    WIP
    """

    config_class = KinovaGen3EndEffectorConfig
    name = "kinova_gen3_end_effector"

    def __init__(self, config: KinovaGen3EndEffectorConfig):
        super().__init__(config)
        self.goal_position = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'theta_x': 0.0, 'theta_y': 0.0, 'theta_z': 0.0}

        self.init_pos = home_position
        self.goal_lock = threading.Lock()
        self.t = threading.Thread(target=self.ee_writer)
        self.config = config
        log("main thread self id: {id(self)}")
        log("main thread goal_position id: {id(self.goal_position)}")

        self.last_goal = {}
        self.time2newgoal = 0
        self.time2reachgoal = 0
        
        self.gain_pos = 0.2
        self.gain_rot = 0.5

        self.vel_lim = 0.1
        self.angular_vel_lim = 5


    def ee_writer(self):
        log("ee_writer self id: {id(self)}")
        log("ee_writer goal_position id: {id(self.goal_position)}")
        log(f"Importing utilities")
        from . import utilities
        log(f"{utilities.__file__}")
        args = utilities.parseConnectionArguments()
        log(f"{args}")
        try:
            with utilities.DeviceConnection.createTcpConnection(args) as router:

            # Create required services
                self.base = BaseClient(router)
                self.base_cyclic = BaseCyclicClient(router)
                
                # Example core
                
                

               
                success = True
                log(f"Moving to home position")
                success &= self.move_to_home_position()

                #self.test_twists()
                input("Continue?")

                try:
                    log("ee_writer goal_position id: {id(self.goal_position)}")
                    log(f"Starting action writer loop")
                    while True:
                        loop_start = time.perf_counter()                        
                        log(f"[ee_writer] goal position {self.goal_position}")
                        success &= self.send_cartesian(self.goal_position)
                        dt_s = time.perf_counter() - loop_start
                        #busy_wait(1 / 30 - dt_s)
                except KeyboardInterrupt:
                    log(f"Exiting action thread")
                    self.running = False
                    raise KeyboardInterrupt("Exited")
        except KeyboardInterrupt:
            return
        except Exception as e:
            log(f"Error {e}. could not initialize.")
    def connect(self, calibrate=False):
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

    def test_twists(self):
        
        input("thetay and -z")
        twist_cmd = Base_pb2.TwistCommand()
        twist_cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        twist = twist_cmd.twist
        twist.linear_x = 0
        twist.linear_y = 0
        twist.linear_z = -0.1
        twist.angular_x = 0
        twist.angular_y = 45
        twist.angular_z = 0
        self.base.SendTwistCommand(twist_cmd)
        time.sleep(1)
        self.base.Stop()
        input("thetax")
        twist_cmd = Base_pb2.TwistCommand()
        twist_cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        twist = twist_cmd.twist
        twist.linear_x = 0
        twist.linear_y = 0
        twist.linear_z = 0
        twist.angular_x = 45
        twist.angular_y = 0
        twist.angular_z = 0
        self.base.SendTwistCommand(twist_cmd)
        time.sleep(1)
        self.base.Stop()
        input("thetaz")
        twist_cmd = Base_pb2.TwistCommand()
        twist_cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        twist = twist_cmd.twist
        twist.linear_x = 0
        twist.linear_y = 0
        twist.linear_z = 0
        twist.angular_x = 0
        twist.angular_y = 0
        twist.angular_z = 45
        self.base.SendTwistCommand(twist_cmd)
        time.sleep(1)
        self.base.Stop()

    def get_present_pose(self):
        feedback = self.base_cyclic.RefreshFeedback()
        return {
            "x": feedback.base.tool_pose_x,
            "y": feedback.base.tool_pose_y,
            "z": feedback.base.tool_pose_z,
            "theta_x": feedback.base.tool_pose_theta_x,
            "theta_y": feedback.base.tool_pose_theta_y,
            "theta_z": feedback.base.tool_pose_theta_z,
        }
    
    def send_cartesian(self, goal_position):
        try:
            if not goal_position == self.last_goal:
                #log(f"T goal update: {time.time() - self.time2newgoal}")
                self.last_goal = goal_position.copy()
                
            self.time2reachgoal = time.time()

            present_pos = self.get_present_pose()
            
            goal = goal_position.copy()
            log(f"[send_cartesian] goal position {goal}")
            
            #log("Starting Cartesian action movement ...")
            try:
                twist_cmd = Base_pb2.TwistCommand()
                twist_cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
                
                target_global_pose = {
                    "x": self.init_pos["x"] + goal["x"],
                    "y": self.init_pos["y"] + goal["y"],
                    "z": self.init_pos["z"] + goal["z"],
                    "theta_x": self.init_pos["theta_x"] + goal["theta_x"],
                    "theta_y": self.init_pos["theta_y"] + goal["theta_y"],
                    "theta_z": self.init_pos["theta_z"] + goal["theta_z"]  
                }
                log(f"Target_global_pose:\n{target_global_pose}")
                
                log(f"Present pose:\n{present_pos}")
                error = {
                    "x": target_global_pose["x"] - present_pos["x"],
                    "y": target_global_pose["y"] - present_pos["y"],
                    "z": target_global_pose["z"] - present_pos["z"],
                    "theta_x": self.wrap_deg(target_global_pose["theta_x"] - present_pos["theta_x"]),
                    "theta_y": self.wrap_deg(target_global_pose["theta_y"] - present_pos["theta_y"]),
                    "theta_z": self.wrap_deg(target_global_pose["theta_z"] - present_pos["theta_z"]),
                } # convert from degrees to twist's expected radians
                
                log(f"Error:\n{error}")
                twist = twist_cmd.twist
                twist.linear_x = self.clamp_linear(error["x"] * self.gain_pos)
                twist.linear_y = self.clamp_linear(error["y"] * self.gain_pos)
                twist.linear_z = self.clamp_linear(error["z"] * self.gain_pos)
                twist.angular_x = self.clamp_angular(error["theta_x"] * self.gain_rot)
                twist.angular_y = self.clamp_angular(error["theta_y"] * self.gain_rot)
                twist.angular_z = self.clamp_angular(error["theta_z"] * self.gain_rot)
                log(f"Sending twist: {twist_cmd}")
                self.base.SendTwistCommand(twist_cmd)
                log(f"Twist sent: {twist_cmd}")
                pos_norm = (error["x"]**2 + error["y"]**2 + error["z"]**2)**0.5
                rot_norm = (error["theta_x"]**2 + error["theta_y"]**2 + error["theta_z"]**2)**0.5
                log(f"Distance to goal: {pos_norm}, rotation to goal {rot_norm}")
                if pos_norm < 0.01 and rot_norm < 0.1:
                    log("Reached goal - stopping")
                    self.base.Stop()
                for i in range(0, 10):
                    log(f"{self.get_present_pose()}")
                    time.sleep(0.1)
                self.base.Stop()
            except Exception as e:
                log(f"{e}")
            self.time2newgoal = time.time()
            dt_s = time.perf_counter() - self.time2reachgoal
            return True
        except Exception as e:
            log(f"Error: {e}")
            raise Exception("Action writer thread failed!")
        
    def clamp_angular(self, vel):
        return max(-self.angular_vel_lim, min(vel, self.angular_vel_lim))
                   
    def clamp_linear(self, vel):
        return max(-self.vel_lim, min(vel, self.vel_lim))

    def send_action(self, action):
        if action["theta_y"] > 0.01:
            log(f"Setting goal position (delta) to {action}")
        #with self.goal_lock:
        self.goal_position["x"] = action["x"]
        self.goal_position["y"] = action["y"]
        self.goal_position["z"] = action["z"]
        self.goal_position["theta_x"] = action["theta_x"]
        self.goal_position["theta_y"] = action["theta_y"]
        self.goal_position["theta_z"] = action["theta_z"]
        self.goal_position["gripper"] = action["gripper"]

        #log(f"[send_action] goal position {self.goal_position} (ID: {id(self.goal_position)})")
        
        return self.goal_position
    
    def wrap_deg(self, angle):
        return (angle + 180) % 360 - 180
    
    def check_for_end_or_abort(self, e):
        """Return a closure checking for END or ABORT notifications

        Arguments:
        e -- event to signal when the action is completed
            (will be set when an END or ABORT occurs)
        """
        def check(notification, e = e):
            #log("EVENT : " + \Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event == Base_pb2.ACTION_END \
            or notification.action_event == Base_pb2.ACTION_ABORT:
                e.set()
        return check

    def send_gripper_command(self, value: float):
        """
            value: float from 0 to 1 corresponding to the trigger value on the VR controller
        """
        target_position = value * 100
        self.base_command = BaseCyclic_pb2.Command()
        self.base_command.frame_id = 0
        self.base_command.interconnect.command_id.identifier = 0
        self.base_command.interconnect.gripper_command.command_id.identifier = 0

        while True:
            try:
                base_feedback = self.base_cyclic.Refresh(self.base_command)

                # Calculate speed according to position error (target position VS current position)
                position_error = target_position - base_feedback.interconnect.gripper_feedback.motor[0].position

                # If positional error is small, stop gripper
                if abs(position_error) < 1.5:
                    position_error = 0
                    self.motorcmd.velocity = 0
                    self.base_cyclic.Refresh(self.base_command)
                    return True
                else:
                    self.motorcmd.velocity = 2.0 * abs(position_error) # "proportional gain" is hard-coded
                    if self.motorcmd.velocity > 100.0:
                        self.motorcmd.velocity = 100.0
                    self.motorcmd.position = target_position

            except Exception as e:
                log("Error in refresh: " + str(e))
                return False
            time.sleep(0.001)
        return True

    def move_to_home_position(self):
        # Make sure the arm is in Single Level Servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)
        
        # Move arm to ready position
        log("Moving the arm to a safe position")
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == "Home":
                action_handle = action.handle

        if action_handle == None:
            log("Can't reach safe position. Exiting")
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
            log("Safe position reached")
        else:
            log("Timeout on action notification wait")
        return finished
    