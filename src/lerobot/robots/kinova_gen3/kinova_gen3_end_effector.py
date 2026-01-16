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
from scipy.spatial.transform import Rotation as R
logger = logging.getLogger(__name__)

from ..robot import Robot

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

# Good source:
# https://github.com/vincekurtz/kinova_drake/blob/master/kinova_station/hardware_station.py#L670



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
        self.goal_position = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'theta_x': 0.0, 'theta_y': 0.0, 'theta_z': 0.0, "gripper": 0.0}

        self.init_pos = home_position
        self.goal_lock = threading.Lock()
        self.t = threading.Thread(target=self.ee_writer)
        self.config = config
        log("main thread self id: {id(self)}")
        log("main thread goal_position id: {id(self.goal_position)}")

        self.last_goal = {}
        self.time2newgoal = 0
        self.time2reachgoal = 0

        self.vel_lim = 0.15
        self.angular_vel_lim = 10

        self.feedback = None
        self.last_feedback_refresh = 0
        self.fire_low_level = False

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
                log("ee_writer goal_position id: {id(self.goal_position)}")
                log(f"Starting action writer loop")
                self.feedback = self.base_cyclic.RefreshFeedback()
                self.last_feedback_refresh = time.time()
                while True:
                    while self.fire_low_level:
                        loop_start = time.perf_counter()                        
                        log(f"[ee_writer] goal position {self.goal_position}")
                        self.send_cartesian(self.goal_position)
                        dt_s = time.perf_counter() - loop_start
                        #busy_wait(1 / 30 - dt_s)
                    time.sleep(0.1)

        except KeyboardInterrupt:
            return
        except Exception as e:
            log(f"Error {e}. could not initialize.")

    def home(self):
        """Returns arm to home position, and (after) opens the low level control lock."""
        self.fire_low_level = False
        time.sleep(0.1) # wait for last twist to finish (otherwise the next line will silently fail)
        self.move_to_home_position()
        self.goal_position["x"] = 0.0
        self.goal_position["y"] = 0.0
        self.goal_position["z"] = 0.0
        self.goal_position["theta_x"] = 0.0
        self.goal_position["theta_y"] = 0.0
        self.goal_position["theta_z"] = 0.0
        self.goal_position["gripper"] = 0.0
        self.fire_low_level = True
        
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
        pass

    def start_low_level(self):
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
        return dict()

    def get_joints_array(self):
        # Read arm position
        joints_arr = []
        for a in self.feedback.actuators:
            position = a.position
            joints_arr.append(position)
        return np.array(joints_arr)

    @property
    def is_calibrated(self):
        return True # CHANGE?

    def disconnect(self):
        pass

    @property
    def is_connected(self) -> bool:
        return self.connected

    def go_to_cartesian_pose(self, pose):
        print("EXPERIMENTAL going to pose...")
        action = Base_pb2.Action()

        action.name = "Cartesian + gripper"

        action.application_data = ""
        cartesian_pose = action.reach_pose.target_pose

        cartesian_pose.x = 0.5766636729240417
        cartesian_pose.y = 0.0013102991
        cartesian_pose.z = 0.436315989494324
        cartesian_pose.theta_x = 90.0
        cartesian_pose.theta_y = 0.0
        cartesian_pose.theta_z = 0.0

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(self.check_for_end_or_abort(e), Base_pb2.NotificationOptions())
        self.base.ExecuteAction(action)
        
        finished = e.wait(TIMEOUT_DURATION)
        self.base.Unsubscribe(notification_handle)

    def experiment_pose(self):
        self.go_to_cartesian_pose({
            "x": 0.57,
            "y": 0.001,
            "z": 90,
            "y": 0,
            "z": 0,
        })

    def get_present_pose(self):
        if time.time() - self.last_feedback_refresh > 0.01:
            self.feedback = self.base_cyclic.RefreshFeedback()
            self.last_feedback_refresh = time.time()

        return {
            "x": self.feedback.base.tool_pose_x,
            "y": self.feedback.base.tool_pose_y,
            "z": self.feedback.base.tool_pose_z,
            "theta_x": self.feedback.base.tool_pose_theta_x,
            "theta_y": self.feedback.base.tool_pose_theta_y,
            "theta_z": self.feedback.base.tool_pose_theta_z,
        }
    
    def get_gripper_value(self):
        # Position is 0 full open, 1 fully closed
        gripper_request = Base_pb2.GripperRequest()
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
        return gripper_measure.finger[0].value
    
    def send_cartesian(self, goal_position):
        try:
            goal = goal_position.copy()
            log(f"[send_cartesian] anchored goal (Euler) {goal}")
            target_global_pose = {
                "x": self.init_pos["x"] + goal["x"],
                "y": self.init_pos["y"] + goal["y"],
                "z": self.init_pos["z"] + goal["z"],
                "theta_x": self.init_pos["theta_x"] + goal["theta_x"],
                "theta_y": self.init_pos["theta_y"] + goal["theta_y"],
                "theta_z": self.init_pos["theta_z"] + goal["theta_z"],
                "gripper": goal["gripper"]
            }
            log(f"Target_global_pose (Euler):\n{target_global_pose}")

            present_pos = self.get_present_pose()
            log(f"Present pose (Euler):\n{present_pos}")

            error = {
                "x": target_global_pose["x"] - present_pos["x"],
                "y": target_global_pose["y"] - present_pos["y"],
                "z": target_global_pose["z"] - present_pos["z"],
                "theta_x": target_global_pose["theta_x"] - present_pos["theta_x"],
                "theta_y": target_global_pose["theta_y"] - present_pos["theta_y"],
                "theta_z": target_global_pose["theta_z"] - present_pos["theta_z"],
            }
            log(f"Error (Euler):\n{error}")

            
            """
            # --- Hypothesis 1: Twist coordinates = global Euler coordinates. Cannot come up with alternative.
            self.apply_twist({
                "linear_x": self.clamp_linear(error["x"] * self.gain_pos),
                "linear_y": self.clamp_linear(error["y"] * self.gain_pos),
                "linear_z": self.clamp_linear(error["z"] * self.gain_pos),
                "angular_x": self.clamp_angular(error["theta_x"] * self.gain_rot), # degrees, as expected
                "angular_y": self.clamp_angular(error["theta_y"] * self.gain_rot), # degrees, as expected
                "angular_z": self.clamp_angular(error["theta_z"] * self.gain_rot) # degrees, as expected
            })
            """
            # --- Hypothesis 2: Twist rotation coordinates are "additive"
            twist_vel = self.compute_twist(present_pos, target_global_pose)
            self.apply_twist(twist_vel)
            """
            # --- Hypothesis 1: Send gripper as error
            current_gripper = self.get_gripper_value()
            gripper_vel = self.compute_gripper(target_global_pose["gripper"], current_gripper)
            self.send_gripper_command(gripper_vel) # outdated func call
            """

            # --- Hypothesis 2: Just set position
            self.send_gripper_command(target_global_pose["gripper"])


                
                     
        except Exception as e:
            log(f"Error: {e}")
            raise Exception("Action writer thread failed!")
        
    def compute_gripper(self, target_gripper, current_gripper, deadband=0.1):
        error = target_gripper - current_gripper
        if abs(error) < deadband:
            return_ = 0.0
        elif error > 0:
            return_ = -1.0
        else:
            return_ = 1.0
        log(f"[gripper] target {target_gripper} - {current_gripper} => {return_}")
        return return_

    def compute_twist(self, current_pose, target_pose, gain_pos=0.8, gain_rot=0.8):
        """
        Compute Twist command to move from current_pose to target_pose. IIIIFFF it needs computing...
        
        Args:
            current_pose: dict with keys ["x","y","z","theta_x","theta_y","theta_z"] in meters and degrees
            target_pose: dict with keys ["x","y","z","theta_x","theta_y","theta_z"] in meters and degrees
            gain_pos: proportional gain for linear velocities
            gain_rot: proportional gain for angular velocities
            
        Returns:
            twist_vel: dict with keys ["linear_x","linear_y","linear_z","angular_x","angular_y","angular_z"]
        """
        # --- Linear error (still straightforward)
        error_linear = np.array([
            target_pose["x"] - current_pose["x"],
            target_pose["y"] - current_pose["y"],
            target_pose["z"] - current_pose["z"]
        ])
        
        # --- Rotational error (convert Euler -> quaternion -> axis-angle)
        current_rot = R.from_euler('xyz', [current_pose["theta_x"], current_pose["theta_y"], current_pose["theta_z"]], degrees=True)
        target_rot  = R.from_euler('xyz', [target_pose["theta_x"], target_pose["theta_y"], target_pose["theta_z"]], degrees=True)
        
        error_rot = target_rot * current_rot.inv()   # rotation from current -> target
        rotvec = error_rot.as_rotvec()               # axis-angle representation, magnitude in radians
        
        # Proportional control for angular velocity
        angular_vel = np.degrees(rotvec) * gain_rot

        twist_vel = {
            "linear_x": self.clamp_linear(error_linear[0] * gain_pos),
            "linear_y": self.clamp_linear(error_linear[1] * gain_pos),
            "linear_z": self.clamp_linear(error_linear[2] * gain_pos),
            "angular_x": self.clamp_angular(angular_vel[0]),
            "angular_y": self.clamp_angular(angular_vel[1]),
            "angular_z": self.clamp_angular(angular_vel[2])
        }
        
        return twist_vel

    def apply_twist(self, velocities):
        twist_cmd = Base_pb2.TwistCommand()
        twist_cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        twist = twist_cmd.twist
        twist.linear_x = velocities["linear_x"]
        twist.linear_y = velocities["linear_y"]
        twist.linear_z = velocities["linear_z"]
        twist.angular_x = velocities["angular_x"]
        twist.angular_y = velocities["angular_y"]
        twist.angular_z = velocities["angular_z"]
        self.base.SendTwistCommand(twist_cmd)
        log(f"Twist sent:\n{twist_cmd}")

    def rotation_matrices(self, pose):
        return self.euler_xyz_deg_to_rotmat(
            pose["theta_x"],
            pose["theta_y"],
            pose["theta_z"]
        )

    def send_gripper_command(self, value):
        """
        Send a position or a velocity command to the gripper
        """

        cmd = Base_pb2.GripperCommand()
        cmd.mode = Base_pb2.GRIPPER_POSITION

        finger = cmd.gripper.finger.add()
        finger.finger_identifier = 0
        finger.value = value

        self.base.SendGripperCommand(cmd)
        
    def rotation_distance(self, R_curr, R_goal):
        R_err = R_goal @ R_curr.T
        return np.arccos(
            np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
        )

    def calculate_angular_twist(self, R_c, R_g):
        # --- Rotation matrices ---
        log(f"Present rotation:\n{R_c}")
        log(f"Goal rotation:\n{R_g}")
        # --- Rotation error in base frame ---
        R_err = R_g @ R_c.T

        # --- Axis–angle ---
        trace = np.trace(R_err)
        angle = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))

        if angle < 1e-6:
            omega = np.zeros(3)
        else:
            axis = np.array([
                R_err[2,1] - R_err[1,2],
                R_err[0,2] - R_err[2,0],
                R_err[1,0] - R_err[0,1],
            ]) / (2 * np.sin(angle))

            omega = self.gain_rot * axis * angle

        # --- Clamp ---
        norm = np.linalg.norm(omega)
        if norm > self.angular_vel_lim:
            omega *= self.angular_vel_lim / norm

        omega_deg = np.rad2deg(omega)
        return omega_deg

    def euler_xyz_deg_to_rotmat(self, theta_x, theta_y, theta_z):
    # Extrinsic XYZ
        r = R.from_euler('xyz', [theta_x, theta_y, theta_z], degrees=True)
        return r.as_matrix()

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
    def test(self):
        self.base.Stop()
        time.sleep(0.2)

        servo = Base_pb2.ServoingModeInformation()
        servo.servoing_mode = Base_pb2.LOW_LEVEL_SERVOING
        self.base.SetServoingMode(servo)
        time.sleep(0.1)

        twist = Base_pb2.TwistCommand()
        twist.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        twist.twist.linear_z = 0.05

        for _ in range(200):
            self.base.SendTwistCommand(twist)
            time.sleep(0.01)

        self.base.Stop()
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
            print("Failed? Going to experiment pose...")
            self.experiment_pose()
            print("Timeout on action notification wait")
        return finished
    
