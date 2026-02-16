# --- Standard library ---
import logging
import os
import random
import shutil
import time
from enum import Enum
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from threading import Thread
import json
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import nullcontext
# --- Third-party ---
import cv2
import numpy as np
import rerun as rr
import torch
import PIL

# --- LeRobot: robots ---
from lerobot.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
    so101_follower,
    kinova_gen3,
)
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.robots.kinova_gen3 import KinovaGen3EndEffectorConfig, KinovaGen3EndEffector
# --- LeRobot: teleoperators ---
from lerobot.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.teleoperators.keyboard.configuration_keyboard import (
    KeyboardJointTeleopConfig,
    KeyboardEndEffectorTeleopConfig,
)
from lerobot.teleoperators.unity.configuration_unity import (
    UnityEndEffectorTeleopConfig,
)

# --- LeRobot: teleoperate ---
from lerobot.teleoperate import (
    TeleoperateConfig,
)

# --- LeRobot: policies ---
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# --- LeRobot: datasets ---
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager

# --- LeRobot: utils ---
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

# Assorted:
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import threading
import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.scripts.eval import eval_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.utils.wandb_utils import WandBLogger

from lerobot.scripts.train import update_policy
import sys
custom_brains = Path("/home/olin/Robotics/Projects/LeRobot/lerobot/custom_brains")#import editable lerobot for VLA
sys.path.append(custom_brains.as_posix())
print(sys.path)
from camera_readers import WebcamReader, USBCameraReader

logger = logging.getLogger(__name__)

import utils

# ======================================= #
#   Helpers / stand-ins for more formalisms
# ======================================= #

class VisionType(Enum):
    KINOVA_HRILAB = "kinova_hrilab"
    SO101_MULIP = "so101_mulip"
    SO101_EYE = "so101_eye"
    NONE = "none"

def sensory_factory_function(vision_type: VisionType) -> dict[str, Any]:
    print(f"Creating vision of type {vision_type}")
    match vision_type:
        case VisionType.KINOVA_HRILAB:
            ob = WebcamReader.get_cap("rtsp://admin:admin@192.168.1.10/color")
            front = USBCameraReader.get_cap(4)
            ra = {
                "front": USBCameraReader(front),
                "onboard": WebcamReader(ob),
            }
            return ra
        case VisionType.SO101_MULIP:
            up = USBCameraReader.get_cap(2)
            side = USBCameraReader.get_cap(4)
            ra = {
                "up": USBCameraReader(up),
                "side": USBCameraReader(side),
            }
            return ra
        case VisionType.SO101_EYE:
            eye = USBCameraReader.get_cap(2)
            ra = {
                "onboard": USBCameraReader(eye),
            }
            return ra
        case VisionType.NONE:
            return {}
        case _:
            raise ValueError(f"Unsupported vision type: {vision_type}")

def create_dataset(robot, teleop_config, dataset_features, dataset_name):
    return LeRobotDataset.create(
        repo_id="olingoudey/" + dataset_name,
        fps=30,
        root=Path('./data/' + dataset_name + str(random.randint(0, 1000))), # random numbers so no datais overridden
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=1,
        image_writer_threads=4 * 2,
        batch_encoding_size=16,
    )

def get_dataset_features(robot, camera_assignments):
    cameras = list(camera_assignments.values())
    robot.set_external_cameras(camera_assignments) # A way to put external cameras in robot attributes
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}
    print(f"Expected dataset features: {dataset_features}")
    return dataset_features   

def start_cameras(cameras_assignments):
    cameras = list(cameras_assignments.values())
    for camera in cameras:
        if camera.is_alive():
            continue
        camera.start()
        while camera.frame is None:
            print(f"Waiting on {str(camera)}...", end="\r")
            time.sleep(0.01)

def create_body(subclass:Robot):
    """
    LeRobot Robot by class -> connected LeRobot Robot (and its confg)
    """
    match subclass.__name__:
        case "SO101Follower":
            try:
                robot_config = SO101FollowerConfig(
                    port="/dev/ttyACM0",
                    id="normal",
                    use_degrees=False,
                )
                robot = SO101Follower(robot_config)
                robot.connect()
            except Exception:
                raise NoRobotException("Could not esablish connection with robot")
        case "KinovaGen3EndEffector":
            try:
                robot_config = KinovaGen3EndEffectorConfig(
                    #stuff
                )
                robot = KinovaGen3EndEffector(robot_config)
                print(f"Robot created. Connecting...")
                robot.connect()
            except Exception:
                raise NoRobotException("Could not esablish connection with robot")
    return robot, robot_config

def create_teleop(robot_config: SO101FollowerConfig, cls: UnityEndEffectorTeleopConfig | KeyboardEndEffectorTeleopConfig):
    """
    TeleopConfig class -> created Teleop config instance (this is really a class method?)
    """
    match cls.__name__:
        case "UnityEndEffectorTeleopConfig": # might not work...
            return UnityEndEffectorTeleopConfig(
                fps=30,
                teleop_time_s=180.0,
                display_data=False,
            )
        case "KeyboardEndEffectorTeleopConfig":
            return KeyboardEndEffectorTeleopConfig(
                id="teleop1",
                calibration_dir=Path("/home/olin/Robotics/Projects/LeRobot/lerobot/custom_brains"),
                mock=False,                
            )
        case _:
            raise Exception(f"Please provide a known Teleop class, not {cls}")


# ======================================= #
#  VLA* Factory <--> LeRobot Interactions #
# ======================================= #


def interloop_log(msg):
    if False:
        print(msg)

class Runner:
    """
    For all the uses you imagine, open values in create()
    """
    def __init__(self):
        """
        Strong defaulting, mostly to test SmolVLA 
        """
        
        self.robot: Robot = None # Needs

        self.repeat_on_episode_end = False
        self.reset_position_on_begin = False
        self.ask_to_reset = False
        self.ask_to_loop = False
        self.demoed = False
        self.teleop_cfg = None
        self.calculate_ik = True
        self.dataset_making = False
        self.dataset_name = "default"
        self.ask_catch_on_end = True
        self.ask_to_save_episode = True
        self.camera_assignments = None
        self.project_camera = False # changed to a str later??!
        self.policy = None
        self.device = None

        self.active_teleop = None

    def run(self, signal):
        """
        Universal run method. Pass a signal that the VLA Complex alters, or that is altered in execute()
        """
        while True: # actual run loop
            try:
                if signal["RUNNING_LOOP"]:
                    try:
                        print(f"In running loop with {signal}")
                        start_cameras(self.camera_assignments)
                        if self.dataset_making:
                            try:
                                dataset_features = get_dataset_features(self.robot, self.camera_assignments)
                                dataset = create_dataset(self.robot, self.teleop_cfg, dataset_features, self.dataset_name)
                                print("Dataset created.")
                            except Exception as e:
                                print(f"Dataset could not be created: {e}")     
                        if self.demoed:
                            if self.active_teleop is None:
                                self.active_teleop = make_teleoperator_from_config(self.teleop_cfg)
                            print(f"Connecting teleop...")
                            self.active_teleop.connect(signal)
                            print(f"Teleop done connecting.")
                            self.active_teleop.send_message(f"Testing teleop...")

                        if not self.robot.is_connected: # Since this loop kills the robot, while its created in the factory
                            self.robot.connect()


                        if hasattr(self.robot, "start_low_level"): # workaround
                            self.robot.start_low_level() # starts thread to actuators

                        if self.reset_position_on_begin:
                            if self.ask_to_reset:
                                self.active_teleop.send_message(f"Reset position?") # Add display_text...
                                while not signal["DECISION"]:
                                    time.sleep(0.1)
                                if signal["DECISION"] == "y":
                                    self.active_teleop.send_message(f"Resetting position...")
                                    self.robot.reset_position()
                                    self.active_teleop.send_message(f"Position reset. Resetting signal...")
                                else:
                                    self.active_teleop.send_message(f"Not resetting position...")
                                signal["DECISION"] = None
                            else:
                                self.active_teleop.send_message(f"Resetting position...")
                                self.robot.reset_position()
                                self.active_teleop.send_message(f"Position reset. Resetting signal...")
                    except Exception as e:
                        print(f"Setup failed: {e}")
                    ctx = VideoEncodingManager(dataset) if self.dataset_making else nullcontext() # For cleanliness...
                    with ctx:
                    
                        while signal["RUNNING_LOOP"]:
                            if self.ask_to_loop:
                                self.active_teleop.send_message(f"Start episode/Quit?")
                                while not signal["DECISION"]:
                                    time.sleep(0.1)
                                if signal["DECISION"] == "n":
                                    self.active_teleop.send_message(f"Quitting!")
                                    signal["RUNNING_LOOP"] = False
                                    signal["DECISION"] = None
                                    self.active_teleop.send_message(f"{signal}")
                                    break
                                else:
                                    self.active_teleop.send_message(f"Go! {signal}")
                                signal["DECISION"] = None
                            
                            try:
                                if self.calculate_ik:
                                    initial_joints_deg = np.array(self.robot.get_joints_array())    # convert to np_array for kinematics
                                    position_weight, orientation_weight = 1.0, 0.1    
                                    calculated_ee_pos = self.active_teleop.kinematics.forward_kinematics(initial_joints_deg)
                                    self.active_teleop.reset(calculated_ee_pos)
                                    self.active_teleop.kinematics.robot.update_kinematics()
                            except Exception as e:
                                self.active_teleop.send_message(f"Error in pre-episode decisions: {e}")
                            self.active_teleop.send_message("Beginning episode loop...")
                            while signal["RUNNING_E"]:
                                interloop_log("Loop start.")
                                loop_start = time.perf_counter()
                                state = self.robot.get_joints_array() # Abstraction
                                observation_frame = {"state": state}
                                for angle, reader in self.camera_assignments.items():
                                    observation_frame[f"observation.images.{angle}"] = reader
                                interloop_log("Got observation.")

                                if self.demoed:
                                    action = self.active_teleop.get_action()
                                    if signal["DECISION"] == "n":
                                        print(signal)
                                        signal["RUNNING_E"] = False
                                        signal["DECISION"] = None
                                        print(signal)
                                else:
                                    action = predict_action(
                                        observation_frame,
                                        self.policy,
                                        device=self.device,
                                        use_amp=(self.device.type == "cuda"),
                                        task=signal["task"],
                                        robot_type=self.robot.robot_type,
                                    )
                                interloop_log("Got input action.")
                                if self.calculate_ik:
                                    try:
                                        target_ee_pos = np.array([action["x"], action["y"], action["z"]])
                                        calculated_ee_pos[:3, 3] = target_ee_pos
                                        target_pitch = np.deg2rad(action["pitch"])   # in degrees
                                        target_roll = np.deg2rad(action["roll"])
                                        R_new = self.active_teleop.rot_y(target_pitch) @ self.active_teleop.rot_z(target_roll)
                                        calculated_ee_pos[:3, :3] = R_new
                                        calculated_new_joints_deg = self.active_teleop.kinematics.inverse_kinematics(state, calculated_ee_pos, position_weight, orientation_weight)
                                        target_gripper = action["gripper"]
                                        action = {name + '.pos': float(val) for name, val in zip(self.active_teleop.joint_names, calculated_new_joints_deg)} # convert back to action dict
                                        action["gripper.pos"] = target_gripper
                                    except Exception as e:
                                        print(f"Error in calculating IK: {e}")
                                    interloop_log("Got raw action.")


                                interloop_log(f"Action: {action}")
                                self.robot.send_action(action)

                                if self.dataset_making:
                                    frame = {
                                        "observation.state": np.array(state, dtype=np.float32),   # robot state
                                        "action": np.array(list(action.values()), dtype=np.float32)
                                    }
                                    for angle, reader in self.camera_assignments.items():
                                        frame[f"observation.images.{angle}"] = reader.frame.copy()
                                        if self.project_camera == angle:
                                            self.active_teleop.project(reader.frame.copy())
                                        else:
                                            pass
                                     
                                    dataset.add_frame(
                                        frame,
                                        task=signal["task"],        # or whatever
                                    )
                                    print(f"Frame off: {frame}")
                                interloop_log(f"End of loop.")
                                dt_s = time.perf_counter() - loop_start
                                busy_wait(1 / 30 - dt_s) # fps is hard-coded  
                                interloop_log(f"Last line of loop.")  
                            if self.dataset_making:
                                if self.ask_to_save_episode:
                                    self.active_teleop.send_message(f"Save episode?")
                                    time.sleep(0.1)
                                    while signal["DECISION"] is None:
                                        time.sleep(0.1)
                                    if signal["DECISION"] == "y":
                                        dataset.save_episode()
                                    else:
                                        self.active_teleop.send_message(f"Not saving")
                                    signal["DECISION"] = None
                                else:
                                    dataset.save_episode()
                            if self.repeat_on_episode_end:
                                signal["RUNNING_E"] = True
                                if self.reset_position_on_begin:
                                    if self.ask_to_reset:
                                        self.active_teleop.send_message(f"Reset position?")
                                        while not signal["DECISION"]:
                                            time.sleep(0.1)
                                        if signal["DECISION"] == "y":
                                            self.active_teleop.send_message(f"Resetting position...")
                                            self.robot.reset_position()
                                        else:
                                            self.active_teleop.send_message(f"Not resetting position...")
                                        signal["DECISION"] = None
                                    else:       

                                        self.active_teleop.send_message(f"Resetting position...")
                                        self.robot.reset_position() # Abstraction for robots, should maybe take an arg, or actually be another VLA
                                        self.active_teleop.send_message(f"Position reset.")
                            else:
                                signal["RUNNING_LOOP"] = False
                        if self.ask_catch_on_end:
                            if self.robot.is_connected:                    
                                self.active_teleop.send_message("Drop? (Catch me!)")
                                while signal["DECISION"] is None:
                                    time.sleep(0.1)
                                if signal["DECISION"] == "y":
                                    for t in range(60, 0, -1):
                                        print(f"\rDropping in...! {t/20:.1f}s", end="", flush=True)
                                        time.sleep(0.05)
                                    self.robot.bus.disable_torque()
                                    self.robot.disconnect() 
                                signal["DECISION"] = None
                        if self.active_teleop:
                            self.active_teleop.send_message("No longer sending teleop...")                    
            except Exception as e:
                print(f"Error in run loop {e}")
                
            time.sleep(1)


def factory_function(vla_complex_cfg) -> Runner:
    """
    pragmatic implicational values
    """
    runner = Runner()
    print(f"LeRobot creating {vla_complex_cfg}...")
    match vla_complex_cfg.agency_type.value:
        case "arm_vr_demo":
            robot, robot_config = create_body(KinovaGen3EndEffector) # defaults to Kinova
            camera_assignments = sensory_factory_function(VisionType.KINOVA_HRILAB)
            teleop_cfg = create_teleop(robot_config, UnityEndEffectorTeleopConfig)

            runner.robot = robot
            runner.repeat_on_episode_end = True
            runner.reset_position_on_begin = True
            runner.ask_to_reset = True
            runner.demoed = True
            runner.teleop_cfg = teleop_cfg
            runner.calculate_ik = False
            runner.camera_assignments = camera_assignments
            runner.project_camera = "onboard"
            runner.ask_to_loop = True
            runner.ask_catch_on_end = False
        case "keyboard_demo":
            runner.robot, robot_config = create_body(SO101Follower)
            runner.camera_assignments = sensory_factory_function(VisionType.SO101_EYE)
            runner.teleop_cfg = create_teleop(robot_config, KeyboardEndEffectorTeleopConfig)
            runner.demoed = True
            runner.calculate_ik = True # redundant
            

            runner.repeat_on_episode_end = True
            runner.reset_position_on_begin = True
            runner.ask_to_reset = True
            runner.ask_to_loop = True
            
        case "auto":
            print(f"LeRobot creating {vla_complex_cfg}...")
            match vla_complex_cfg.robot_type.value:
                case "kinova":
                    pass
                    # not sure how to instantiate a VLA for Kinova
                    #   It could have all the functions that are below for so101
                    # 
                    #
                    #
                    #
                case "so101":
                    runner.policy = SmolVLAPolicy.from_pretrained(vla_complex_cfg.policy_path)
                    runner.camera_assignments = sensory_factory_function(VisionType.SO101_MULIP)
                    if torch.cuda.is_available():
                        print("Running CUDA")
                        runner.device = torch.device("cuda")
                    else:
                        print(f"Initializing weak brain...")
                        runner.device = torch.device("cpu")
                    runner.calculate_ik = False
                case _:
                    raise ValueError(f"Cannot create AUTO VLA Complex from {vla_complex_cfg}")
        case _:
            raise ValueError(f"Cannot create VLA Complex from {vla_complex_cfg}")
    
    runner.dataset_making = vla_complex_cfg.recorded
    if vla_complex_cfg.recorded:
        if not vla_complex_cfg.dataset_name:
            raise ValueError(f"Must provide a dataset name in main config")
        runner.dataset_name = vla_complex_cfg.dataset_name 

    if not runner:
        raise ValueError("Could not create Runner!")
    print(f"Runner created.")
    return runner