# --- Standard library ---
import logging
import os
import random
import shutil
import time
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
    teleop_loop,
    teleop_loop_no_ik,
    mock_teleop_loop,
    unrecorded_teleop_loop,
    unrecorded_teleop_loop_no_ik,
    no_robot_loop,
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

def create_dataset(robot, teleop_config, dataset_features, dataset_name):
    return LeRobotDataset.create(
        repo_id="olingoudey/" + dataset_name,
        fps=teleop_config.fps,
        root=Path('./data/' + dataset_name + str(random.randint(0, 1000))), # random numbers for safety
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * 2,
        batch_encoding_size=16,
    )

def episode_run(robot, teleop_config, reader_assignments, task_name, dataset, with_ik, signal: dict[str, bool]):
    teleop = make_teleoperator_from_config(teleop_config)

    if with_ik:
        teleop_loop(teleop, robot, teleop_config.fps, display_data=teleop_config.display_data, duration=teleop_config.teleop_time_s, reader_assignments=reader_assignments, dataset=dataset, signal=signal)
    else:
        teleop_loop_no_ik(teleop, robot, teleop_config.fps, duration=teleop_config.teleop_time_s, reader_assignments=reader_assignments, dataset=dataset, signal=signal)

def exit_episode(robot, teleop, dataset):
    input("\nReset robot? (^C to exit)")
    robot = utils.reset_bw_episode(robot, teleop)
    input("[hit Enter]")
    #new_task = input("\nNew task? (hit only Enter to use same task) (^C to exit)") # environment scenario updated manually
    #if new_task:
    #    task = new_task

def run_safely_wrapper(robot, teleop_config, reader_assignments, task_name, dataset, with_ik, signal: dict[str, bool]):
    try:
        episode_run(robot, teleop_config, reader_assignments, task_name, dataset, with_ik, signal)
    except Exception as e:
        print("Robot connection error?:", e)
        print("Reconnecting... (Catch me!)")
        time.sleep(1)
        robot.disconnect()
        robot.connect()
        input("Continue?")

def get_dataset_features(robot, camera_assignments):
    cameras = list(camera_assignments.values())
    robot.external_cameras = [camera.frame.shape for camera in cameras] # A way to put external cameras in robot attributes
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}   
    return dataset_features   

def start_cameras(cameras_assignments):
    cameras = list(cameras_assignments.values())
    for camera in cameras:
        if camera.is_alive():
            continue
        camera.start()
        while camera.frame is None:
            print(f"\rWaiting... on {camera}")
            time.sleep(0.01)


def episode_loop(robot, teleop_config, reader_assignments, dataset, with_ik, signal: dict[str, bool]):
    
    with VideoEncodingManager(dataset):
        while signal["RUNNING_LOOP"]:
            logging.info("New episode starting... ^C when done or to stop.")
            run_safely_wrapper(robot, teleop_config, reader_assignments, signal["task"], dataset, with_ik, signal)
        while not "dataset_name" in signal:
            pass
        dataset.name = signal["dataset_name"]
        exit_episode_loop(robot, teleop_config, reader_assignments, dataset)

def exit_episode_loop(robot, teleop_config, reader_assignments, dataset):
    teleop = make_teleoperator_from_config(teleop_config)
    cameras = list(reader_assignments.values())
    logging.info("\nExiting episode loop.")      
    for camera in cameras:
        camera.stop()
    input("[hit Enter to catch me]\n")
    for t in range(60, 0, -1):
        print(f"\rDropping in...! {t/20:.1f}s", end="", flush=True)
        time.sleep(0.05)
    logging.info("\rBye!      ") 
    
    robot.bus.disable_torque()
    if teleop_config.display_data:
        rr.rerun_shutdown()
    teleop.disconnect()
    robot.disconnect()   

# ======================================= #
#  VLA* Factory <--> LeRobot Interactions #
# ======================================= #

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
        self.ask_catch_on_end = True
        self.camera_assignments = None
        self.project_camera = False # changed to a str later??!

        self.policy = None
        self.device = None

    def run(self, signal, dataset_name: Optional[str]=None):
        """
        Universal run method. Pass a signal that the VLA Complex alters, or that is altered in execute()
        """
        start_cameras(self.camera_assignments)
        if self.dataset_making:
            dataset_features = get_dataset_features(self.robot, self.camera_assignments)
            dataset = create_dataset(self.robot, self.teleop_cfg, dataset_features, dataset_name)
                    
        if self.demoed:
            teleop = make_teleoperator_from_config(self.teleop_cfg)
            teleop.connect(signal)

        if hasattr(self.robot, "start_low_level"): # workaround
            self.robot.start_low_level() # starts thread to actuators  

        ctx = VideoEncodingManager(dataset) if self.dataset_making else nullcontext() # For cleanliness...
        with ctx:
            while signal["RUNNING_LOOP"]:
                if self.reset_position_on_begin:
                    if self.ask_to_reset:
                        print(f"Reset position?") # Add display_text...
                        while not signal["RUNNING_E"]:
                            time.sleep(0.1)
                    print(f"Resetting position.")
                    self.robot.reset_position() # Abstraction for robots, should maybe take an arg, or actually be another VLA
                if self.ask_to_loop:
                    signal["RUNNING_E"] = False
                    print(f"Position reset. Resetting signal...")
                    while not signal["RUNNING_E"]:
                        time.sleep(0.1)

                initial_joints_deg = np.array(self.robot.get_joints_array())    # convert to np_array for kinematics
                if self.calculate_ik:
                    position_weight, orientation_weight = 1.0, 0.1    
                    calculated_ee_pos = teleop.kinematics.forward_kinematics(initial_joints_deg)
                    init_fk = calculated_ee_pos[:3, 3]
                    teleop.target_pos["x"], teleop.target_pos["y"], teleop.target_pos["z"] = init_fk
                    teleop.kinematics.robot.update_kinematics()

                while signal["RUNNING_E"]:
                    loop_start = time.perf_counter()
                    state = self.robot.get_joints_array() # Abstraction
                    observation_frame = {"state": joints_deg}
                    for angle, reader in reader_assignments.items():
                        observation_frame[f"observation.images.{angle}"] = reader

                    
                    if self.demoed:
                        action = teleop.get_action()
                    else:
                        action = predict_action(
                            observation_frame,
                            self.policy,
                            device=self.device,
                            use_amp=(self.device.type == "cuda"),
                            task=signal["task"],
                            robot_type=self.robot.robot_type,
                        )
                    
                    
                    if self.calculate_ik:
                        target_ee_pos = np.array([action["x"], action["y"], action["z"]])
                        calculated_ee_pos[:3, 3] = target_ee_pos
                        target_pitch = np.deg2rad(action["pitch"])   # in degrees
                        target_roll = np.deg2rad(action["roll"])
                        R_new = rot_y(target_pitch) @ rot_z(target_roll)
                        calculated_ee_pos[:3, :3] = R_new
                        calculated_new_joints_deg = teleop.kinematics.inverse_kinematics(state, calculated_ee_pos, position_weight, orientation_weight)
                        target_gripper = action["gripper"]
                        action = {name + '.pos': float(val) for name, val in zip(teleop.joint_names, calculated_new_joints_deg)} # convert back to action dict
                        action["gripper.pos"] = target_gripper

                    
                    self.robot.send_action(action)

                    if self.dataset_making:
                        frame = {
                            "observation.state": np.array(state, dtype=np.float32),   # robot state
                            "action": np.array(list(action.values()), dtype=np.float32)
                        }
                        for angle, reader in self.camera_assignments.items():
                            frame[f"observation.images.{angle}"] = reader.frame.copy()
                            if self.project_camera == angle:
                                teleop.project(reader.frame.copy())
                            else:
                                pass
                        dataset.add_frame(
                            frame,
                            task=signal["task"],        # or whatever
                        )
                    dt_s = time.perf_counter() - loop_start
                    busy_wait(1 / 30 - dt_s) # fps is hard-coded    
            for camera in list(self.camera_assignments.values()):
                camera.stop()
            if self.ask_catch_on_end:
                logging.info("\nExiting episode loop.")      
                
                input("[hit Enter to catch me]\n")
                for t in range(60, 0, -1):
                    print(f"\rDropping in...! {t/20:.1f}s", end="", flush=True)
                    time.sleep(0.05)
                self.robot.bus.disable_torque()
                self.robot.disconnect() 
                if self.demoed:
                    teleop.disconnect()

def factory_function(vla_complex_cfg) -> Runner:
    """
    pragmatic implicational values
    """
    runner = Runner()
    match vla_complex_cfg.agency_type:
        case "arm_vr_demo":
            
            robot, robot_config = create_body(KinovaGen3EndEffector) # defaults to Kinova
            camera_assignments = get_kinova_setup_cameras()
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
            runner.camera_assignments = get_so101_setup_cameras()
            runner.teleop_cfg = create_teleop(robot_config, KeyboardEndEffectorTeleopConfig)
            runner.demoed = True
            runner.calculate_ik = True # redundant
            runner.dataset_making = True
            runner.repeat_on_episode_end = True
            runner.reset_position_on_begin = True
            runner.ask_to_reset = True
            runner.ask_to_loop = True
            
        case "auto":
            match vla_complex_cfg.robot_type:
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
                    runner.camera_assignments = get_so101_setup_cameras()
                    if torch.cuda.is_available():
                        print("Running CUDA")
                        runner.device = torch.device("cuda")
                    else:
                        print(f"Initializing weak brain...")
                        runner.device = torch.device("cpu")
                    runner.calculate_ik = False
                    
    if vla_complex_cfg.recording:
        runner.dataset_making = True    

    if not runner:
        raise ValueError("Could not create Runner!")
    return runner


"""
robot, robot_config = create_body(KinovaGen3EndEffector) # defaults to Kinova
    
    reader_assignments = get_kinova_setup_cameras()
    human_policy: TeleoperateConfig = create_teleop(robot_config, UnityEndEffectorTeleopConfig)

    if dataset_name is None:
        dataset_name = "test_record-11-24"
    return DatasetRecorder(robot, human_policy, dataset_name, reader_assignments)
"""

def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None, reader_assignments: dict[str, Any] = dict(), dataset=None, verbose=False, signal: dict[str, Any]={"RUNNING_LOOP": True, "RUNNING_E": True, "task": ""}
):
    
    if not robot.bus.is_connected: # ignored in offer
        robot.bus.connect()

    display_len = max(len(key) for key in robot.action_features)

    
    position_weight, orientation_weight = 1.0, 0.1
    
    """ Calculate FK once for initial position """
    observation = robot.get_observation() # set robot.present_pos
    initial_joints_deg = np.array([robot.present_pos[name] for name in teleop.joint_names])    # convert to np_array for kinematics
    
    # Check kinematics
    kinematics_joint_order = list(teleop.kinematics.robot.model.names)[2:]
    assert kinematics_joint_order == teleop.joint_names
        
    calculated_ee_pos = teleop.kinematics.forward_kinematics(initial_joints_deg)
    
    init_fk = calculated_ee_pos[:3, 3]
    print(init_fk)

    if type(teleop).__name__ == "KeyboardEndEffectorTeleop":
        teleop.target_pos["x"], teleop.target_pos["y"], teleop.target_pos["z"] = init_fk

    teleop.kinematics.robot.update_kinematics()
            
    start = time.perf_counter()
    print("Teleop loop starting...")
    while signal["RUNNING_E"]:
        loop_start = time.perf_counter()
        
        observation = robot.get_observation()
        joints_deg = np.array([robot.present_pos[name] for name in teleop.joint_names])
        action = teleop.get_action()
        
        if display_data:
            log_rerun_data(observation, action)
        
        if type(teleop).__name__ == "KeyboardEndEffectorTeleop":
            """ Re-Calculate action """
            target_ee_pos = np.array([action["x"], action["y"], action["z"]])
            calculated_ee_pos[:3, 3] = target_ee_pos
            # Now affect R
            target_pitch = np.deg2rad(action["pitch"])   # in degrees
            target_roll = np.deg2rad(action["roll"])
            R_new = rot_y(target_pitch) @ rot_z(target_roll)

            calculated_ee_pos[:3, :3] = R_new
            
            calculated_new_joints_deg = teleop.kinematics.inverse_kinematics(joints_deg, calculated_ee_pos, position_weight, orientation_weight)
            target_gripper = action["gripper"]
            action = {name + '.pos': float(val) for name, val in zip(teleop.joint_names, calculated_new_joints_deg)} # convert back to action dict
            action["gripper.pos"] = target_gripper
        elif type(teleop).__name__ == "UnityEndEffectorTeleop":
            calculated_new_joints_deg = teleop.kinematics.inverse_kinematics(joints_deg, calculated_ee_pos, position_weight, orientation_weight)
            target_gripper = action["gripper"]
            action = {name + '.pos': float(val) for name, val in zip(teleop.joint_names, calculated_new_joints_deg)} # convert back to action dict
            action["gripper.pos"] = target_gripper
            
        robot.send_action(action) # comment for mock?
        
        if dataset is not None:
            frame = {
                "observation.state": np.array(joints_deg, dtype=np.float32)   # robot state
            }
            for angle, reader in reader_assignments.items():
                frame[f"observation.state.{angle}"] = reader.frame.copy()
            frame["action"] = np.array(calculated_new_joints_deg, dtype=np.float32)
            dataset.add_frame(
                frame,
                task=signal["task"],
            )
        
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start
        if verbose:
            logging.info("\n" + "-" * (display_len + 10))

            for motor, value in action.items():
                logging.info(f"{motor:<{display_len}} | {value:>7.2f}")

            logging.info(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(len(action) + 10)
        if duration is not None and time.perf_counter() - start >= duration:
            return




def create_body(type:SO101Follower | KinovaGen3EndEffector=KinovaGen3EndEffector):
    """
    robot / policy / record?
    """
    if type == SO101Follower:
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
    elif type == KinovaGen3EndEffector:
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
    if cls is UnityEndEffectorTeleopConfig:
        return UnityEndEffectorTeleopConfig(
            fps=30,
            teleop_time_s=180.0,
            display_data=False,
        )
    elif cls is KeyboardEndEffectorTeleopConfig:
        return KeyboardEndEffectorTeleopConfig(
            robot = robot_config,
            teleop = KeyboardEndEffectorTeleopConfig(
                id="teleop1",
                calibration_dir=Path("."),
                mock=False,
            ),
            fps=30,
            teleop_time_s=180.0,
            display_data=False,
            
        )
    else:
        raise Exception(f"Please provide a known Teleop class, not {cls}")

def create_raw_teleop():
    unity_teleop: TeleoperateConfig = create_teleop(None, UnityEndEffectorTeleopConfig)
    return RawTeleopRunner(unity_teleop)

def create_raw_teleop_mock(use_laptop_camera=False):
    unity_teleop: TeleoperateConfig = create_teleop(None, UnityEndEffectorTeleopConfig)
    if use_laptop_camera:
        from camera_readers import WebcamReader, USBCameraReader
        reader_assignments = {
            "onboard": USBCameraReader(USBCameraReader.get_cap(0)),
        }
    else:
        reader_assignments = {}
    return MockRawRunner(unity_teleop, reader_assignments)

def create_teleop_unrecorded_interaction():
    robot, robot_config = create_body()
    reader_assignments = get_kinova_setup_cameras()
    human_policy = create_teleop(robot_config, UnityEndEffectorTeleopConfig)
    return RawTeleopRunner(robot, human_policy, reader_assignments)
    

def create_teleop_recorded_interaction(dataset_name: str | None = None):
    robot, robot_config = create_body(KinovaGen3EndEffector) # defaults to Kinova
    
    reader_assignments = get_kinova_setup_cameras()
    human_policy: TeleoperateConfig = create_teleop(robot_config, UnityEndEffectorTeleopConfig)

    if dataset_name is None:
        dataset_name = "test_record-11-24"
    return DatasetRecorder(robot, human_policy, dataset_name, reader_assignments)

def create_so101_teleop_recording_interaction(dataset_name: str | None = None):
    robot, robot_cfg = create_body(SO101Follower)
    reader_assignments = get_so101_setup_cameras()
    teleop_cfg: TeleoperateConfig = create_teleop(robot_cfg, KeyboardEndEffectorTeleopConfig)
    if dataset_name is None:
        dataset_name = "test_record-11-24"
    return DatasetRecorder(robot, teleop_cfg, dataset_name, reader_assignments)

def create_so101_teleop_recording_interaction(dataset_name: str | None = None):
    robot, robot_cfg = create_body(SO101Follower)
    reader_assignments = get_so101_setup_cameras()
    teleop_cfg: TeleoperateConfig = create_teleop(robot_cfg, KeyboardEndEffectorTeleopConfig)
    if dataset_name is None:
        dataset_name = "test_record-11-24"
    return DatasetRecorder(robot, teleop_cfg, dataset_name, reader_assignments)

def create_teleop_recording_kinova_interaction(reader_assignments: dict | None = None, dataset_name: str | None = None):
    robot, robot_config = create_body()
    robot.start_low_level()
    human_policy: TeleoperateConfig = create_teleop(robot_config, UnityEndEffectorTeleopConfig)
    if reader_assignments is None:
        from camera_readers import WebcamReader, USBCameraReader
        reader_assignments = {
            "front": USBCameraReader(USBCameraReader.get_cap(6)),
            "onboard": WebcamReader(WebcamReader.get_cap("rtsp://admin:admin@192.168.1.10/color"))
        }
    if dataset_name is None:
        dataset_name = "demo-12-5"
    return DatasetRecorder(robot, human_policy, dataset_name, reader_assignments)

def get_kinova_setup_cameras():
    ob = WebcamReader.get_cap("rtsp://admin:admin@192.168.1.10/color")
    front = USBCameraReader.get_cap(6)
    ra = {
        "front": USBCameraReader(front),
        "onboard": WebcamReader(ob),
    }
    return ra

def get_so101_setup_cameras():
    ob = USBCameraReader.get_cap(2)
    front = USBCameraReader.get_cap(4)
    ra = {
        "front": USBCameraReader(front),
        "onboard": WebcamReader(ob),
    }
    return ra

def main():
    """ A repetoire of useful main functions: """
    #test_webcam("https://192.168.0.159:8080/shot.jpg")
    #test_webcam("https://192.168.0.151:8080/shot.jpg")
    
    #dummy_dataset()

    #merge_datasets("data/fg", "data/g2",  "data/f5")
    #check_episode_stats("data/f5/meta/episodes_stats.jsonl")
    
    # I "outsource" the train script

    #teleoperate(teleop_config())
    #record_dataset(dataset_name="move_mouse", camera_refs=["rtsp://10.243.51.52:8080/h264_ulaw.sdp", "rtsp://10.243.115.110:8080/h264_ulaw.sdp"]) 
    #record_dataset(dataset_name="blocks_box2", camera_refs=[2, 4]) 
    #teleoperate(teleop_config())
    
    #test_policy("/home/mulip-guest/LeRobot/lerobot/outputs/stationary_env_3k/pretrained_model", camera_urls=["rtsp://10.243.112.170:8080/h264_ulaw.sdp", "rtsp://10.243.63.69:8080/h264_ulaw.sdp"])
    

    
    
    #test_policy("/home/mulip-guest/LeRobot/lerobot/outputs/blocks_box/checkpoints/021000/pretrained_model", camera_urls=["rtsp://10.243.59.185:8080/h264_ulaw.sdp", "rtsp://10.243.126.188:8080/h264_ulaw.sdp"])
    
    #f = create_raw_teleop_mock(True)
    #f.run()

    #return


    r = create_teleop_unrecorded_interaction()
    r.run(None)
    
    #dataset_recorder = create_teleop_recording_kinova_interaction()
    #dataset_recorder.record()

if __name__ == "__main__":
    #main_with_signal({"flag":"STOP", "instruction": "STOP")
    main()
