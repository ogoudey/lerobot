# --- Standard library ---
import logging
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from distutils.util import strtobool
from pathlib import Path
from pprint import pformat
from threading import Thread
import json
import shutil
from pathlib import Path
from typing import List

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

def get_dataset_features(robot, cameras):
    robot.external_cameras = [camera.frame.shape for camera in cameras] # A way to put external cameras in robot attributes
    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}   
    return dataset_features   

def record(robot: SO101Follower | KinovaGen3EndEffector, teleop_config: TeleoperateConfig, dataset_name, reader_assignments: dict[str, WebcamReader | USBCameraReader], with_ik=False, signal:dict={"RUNNING_LOOP":True, "RUNNING_E": True, "task":"default_task_name"}):
    cameras = list(reader_assignments.values())
    for camera in cameras:
        if camera.is_alive():
            continue
        camera.start()
        while camera.frame is None:
            print(f"\rWaiting... on {camera}")
            time.sleep(0.01)

    dataset_features = get_dataset_features(robot, cameras)

    dataset = create_dataset(robot, teleop_config, dataset_features, dataset_name)
    episode_loop(robot, teleop_config, reader_assignments, dataset, with_ik, signal)

def episode_loop(robot, teleop_config, reader_assignments, dataset, with_ik, signal: dict[str, bool]):
    teleop = make_teleoperator_from_config(teleop_config)
    teleop.connect()

    input(f"[Enter] to record '{signal['task']}'")
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
        print(f"\r\Dropping in...! {t/20:.1f}s", end="", flush=True)
        time.sleep(0.05)
    logging.info("\rBye!      ") 
    
    robot.bus.disable_torque()
    if teleop_config.display_data:
        rr.rerun_shutdown()
    teleop.disconnect()
    robot.disconnect()   

def test(robot: SO101Follower, reader_assignments: dict[str, WebcamReader | USBCameraReader], device, policy: SmolVLAPolicy, signal={ "flag": "GO", "instruction": "Put the colored blocks in the cardboard box" }):
    _init_rerun(session_name="SmolVLA")

    cameras = list(reader_assignments.values())
    for camera in cameras:
        camera.start()
        while camera.frame is None:
            print(f"\rWaiting... on {camera}")
            time.sleep(0.01)
    robot.external_cameras = [camera.frame.shape for camera in cameras]
    
    logging.info("Cameras on.")
    input("[hit Enter]")
    robot = utils.reset_bw_episode(robot, None)
    instruction = signal["instruction"]
    try:
        
        while True:
            try:
                while not signal["flag"] == "STOP":
                    print(signal)
                    start_loop_t = time.perf_counter()
                    robot.get_observation()
                    printable_state = {name: round(robot.present_pos[name.split(".pos")[0]], 6) for i, name in enumerate(robot.action_features)}
                    #print("STATE:  ", printable_state)
                    joints_deg = np.array([robot.present_pos[name.split(".pos")[0]] for name in robot.action_features], dtype=np.float32)
                    observation_frame = {"state": joints_deg}
                    for angle, reader in reader_assignments.items():
                        observation_frame[f"observation.images.{angle}"] = reader

                    
                    action_values = predict_action(
                        observation_frame,
                        policy,
                        device=device,
                        use_amp=(device.type == "cuda"),
                        task=instruction,
                        robot_type=robot.robot_type,
                    )

                    
                    #logging.info(f"Gripper diff: {robot.present_pos['gripper.pos'] - action_values[5]}")
                    action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}

                    robot.send_action(action)
                    log_rerun_data(observation_frame, action)
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / 30 - dt_s) # this is 1 / fps - dt_s
                if signal["flag"] == "STOP":
                    break
            except KeyboardInterrupt:
                chat = input("New task? (^C to Exit, [Enter] to try the same task again)\n")
                if not chat == "":
                    single_task=chat   # whether this changes anything is unclear
                robot = utils.reset_bw_episode(robot, None)
                policy.reset()
                input("[Enter]")
        print("Leaving model inference...")
        raise KeyboardInterrupt("Essentially a keyboard interrupt")
    except KeyboardInterrupt:
        for camera in cameras:
            camera.stop()
        input("[hit Enter to catch me]\n") 
        for t in range(60, 0, -1):
            print(f"\r\Dropping in...! {t/20:.1f}s", end="", flush=True)
            time.sleep(0.05)
        logging.info("\rBye!      ")
        robot.bus.disable_torque()
        robot.disconnect()



# ================================================== #
#                 Factory Functions                  #
# ================================================== #

class NoRobotException(Exception):
    pass

class VLAInitializationError(Exception):
    pass

# ============ Runners ============ #

class DatasetRecorder:
    """"""
    def __init__(self, robot, policy_config, dataset_name, reader_assignments):
        self.robot = robot
        self.policy_config = policy_config
        self.dataset_name = dataset_name
        self.reader_assignments = reader_assignments
        self.dataset_name = dataset_name

    def run(self, running_signal):
        if type(self.robot) == KinovaGen3EndEffector:
            print(f"Robot {self.robot} is a KinovaGen3EndEffector, not using IK")
            with_ik = False
        else:
            print(f"Robot is {self.robot}.")
            print(f"Robot uses IK")
            with_ik = True
            
        record(self.robot, self.policy_config, self.dataset_name, self.reader_assignments, with_ik, running_signal)
        print(f"Runner done.")

class MockDatasetRecorder:
    """"""
    def __init__(self, policy_config, dataset_name):
        self.policy_config = policy_config
        self.dataset_name = dataset_name

    def run(self, running_signal):
        while running_signal["RUNNING_LOOP"]:
            print(running_signal)
            time.sleep(1)
        print(f"Runner done.")


class RawTeleopRunner:
    def __init__(self, teleop_config):
        self.teleop_config = teleop_config
        
    # import osmething other than teleop_loop
    def run(self, robot, with_ik=False):
        teleop = make_teleoperator_from_config(self.teleop_config)
        teleop.connect()
        input("Ready to enter loop?")
        while True:
            try:
                if with_ik:
                    unrecorded_teleop_loop(teleop, robot, self.teleop_config.fps, self.teleop_config.display_data, self.teleop_config.teleop_time_s, verbose=False) # send IPwebcam to teleop loop 
                else:
                    unrecorded_teleop_loop_no_ik(teleop, robot, 30, 400) # send IPwebcam to teleop loop                     
            except KeyboardInterrupt:
                input("[hit Enter to catch me]\n")
                for t in range(60, 0, -1):
                    print(f"\r\Dropping in...! {t/20:.1f}s", end="", flush=True)
                    time.sleep(0.05)
                logging.info("\rBye!      ") 
                
                robot.bus.disable_torque()
                if self.teleop_config.display_data:
                    rr.rerun_shutdown()
                teleop.disconnect()
                robot.disconnect()
                break   

class MockRawRunner:
    """No robot, no recording. Accepts UnityEndEffectorTeleopConfig."""
    def __init__(self, teleop_config):
        self.teleop_config = teleop_config
    # import osmething other than teleop_loop
    def run(self):
        teleop = make_teleoperator_from_config(self.teleop_config)
        teleop.connect()
        input("Ready to enter loop?")

        while True:
            no_robot_loop(teleop, self.teleop_config.fps, self.teleop_config.teleop_time_s,) # send IPwebcam to teleop loop 

class SmolVLARunner:
    """Runs smolvla."""
    def __init__(self, device, policy: SmolVLAPolicy):
        self.device = device
        self.policy = policy

    def run(self, robot: SO101Follower, reader_assignments):
        test(robot, reader_assignments, self.device, self.policy)

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

def create_raw_teleop_mock():
    unity_teleop: TeleoperateConfig = create_teleop(None, UnityEndEffectorTeleopConfig)
    return MockRawRunner(unity_teleop)

def create_teleop_unrecorded_interaction():
    robot, robot_config = create_body()
    robot.configure()
    human_policy = create_teleop(robot_config, UnityEndEffectorTeleopConfig)
    r = RawTeleopRunner(human_policy)
    r.run(robot)

def create_unrecorded_mock_interaction(dataset_name="mock"):
    human_policy = create_raw_teleop_mock()
    return MockDatasetRecorder(human_policy, dataset_name)

def create_teleop_recording_interaction(reader_assignments: dict | None = None, dataset_name: str | None = None):
    robot, robot_config = create_body()
    robot.configure()
    human_policy: TeleoperateConfig = create_teleop(robot_config, UnityEndEffectorTeleopConfig)
    if reader_assignments is None:
        from camera_readers import WebcamReader, USBCameraReader
        reader_assignments = {
            "side": USBCameraReader(USBCameraReader.get_cap(2)),
            "up": USBCameraReader(USBCameraReader.get_cap(4))
        }
    if dataset_name is None:
        dataset_name = "test_record-11-24"
    return DatasetRecorder(robot, human_policy, dataset_name, reader_assignments)

def create_teleop_recording_kinova_interaction(reader_assignments: dict | None = None, dataset_name: str | None = None):
    robot, robot_config = create_body()
    robot.configure()
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

def create_brains(reader_assignments: dict, policy_path: Path):
    smolvla_policy = SmolVLAPolicy.from_pretrained(policy_path)
    # assert that the policy's angles are such and such
    if torch.cuda.is_available():
        print("Running CUDA")
        device = torch.device("cuda")
    else:
        print(f"Initializing weak brain...")
        device = torch.device("cpu")

    reader_angles = list(reader_assignments.keys())
    
    if "side" in reader_angles and "up" in reader_angles:
        return SmolVLARunner(device, smolvla_policy)
    elif "onboard" in reader_angles:
        VLAInitializationError("Make sure that the policy has on_body observation frame slot")
    else:
        VLAInitializationError("No SmolVLA found for camera angles.")





def main_with_signal(signal):
    if signal["flag"] == "STOP":
        print("Trying to stop!")
    test_policy("/home/mulip-guest/LeRobot/lerobot/outputs/blocks_box/checkpoints/021000/pretrained_model", camera_urls=["rtsp://10.243.59.185:8080/h264_ulaw.sdp", "rtsp://10.243.126.188:8080/h264_ulaw.sdp"], signal=signal)
  
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
    dataset_recorder = create_teleop_recording_kinova_interaction()
    dataset_recorder.record()

if __name__ == "__main__":
    #main_with_signal({"flag":"STOP", "instruction": "STOP")
    main()
