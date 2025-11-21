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
)
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

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
    test_record_loop,
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

from camera_readers import IPWebcamReader, LogitechReader

logger = logging.getLogger(__name__)

def record_dataset(dataset_name="dataset3", camera_refs=["rtsp://192.168.0.159:8080/h264_ulaw.sdp", "rtsp://192.168.0.151:8080/h264_ulaw.sdp"]):
    t_cfg = teleop_config()

    init_logging()
    logging.info(pformat(asdict(t_cfg)))
    if t_cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(t_cfg.teleop)
    robot = make_robot_from_config(t_cfg.robot)
    input(f"Start cameras with references: {camera_refs}... [hit Enter to continue]")

    webcam1_idx = camera_refs[0]
    webcam2_idx = camera_refs[1]
    webcam1_cap = LogitechReader.get_cap(webcam1_idx)
    webcam2_cap = LogitechReader.get_cap(webcam2_idx)
    webcam1_reader = LogitechReader(webcam1_cap)
    webcam2_reader = LogitechReader(webcam2_cap)
    webcam1_reader.start()
    webcam2_reader.start()

    """
    webcam1_url = camera_refs[0]
    webcam2_url = camera_refs[1]
    webcam1_cap = cv2.VideoCapture(webcam1_url)
    webcam2_cap = cv2.VideoCapture(webcam2_url)
    webcam1_reader = IPWebcamReader(webcam1_cap)
    webcam2_reader = IPWebcamReader(webcam2_cap)
    webcam1_reader.start()
    webcam2_reader.start()
    """
    cameras = [webcam1_reader, webcam2_reader]
    print("Giving cameras some time...")
    time.sleep(2)
    if not webcam1_cap.isOpened() or not webcam2_cap.isOpened():
        time.sleep(1)
        print("More time...")
        if not webcam1_cap.isOpened() or not webcam2_cap.isOpened():
            raise RuntimeError("Cannot open IP webcam")
    if not webcam1_cap.isOpened() or not webcam2_cap.isOpened():
        raise RuntimeError("Cannot open IP webcam") 
    while webcam1_reader.frame is None:
        print("\rWaiting... on camera1")
        time.sleep(0.01)
    while webcam2_reader.frame is None:
        print("\rWaiting... on camera2")
        time.sleep(0.01)
    logging.info("Cameras on.")

    webcam1_image_shape = webcam1_reader.frame.shape
    webcam2_image_shape = webcam2_reader.frame.shape
    print("Camera shapes:", webcam1_image_shape, webcam2_image_shape)
    
    robot.external_cameras = [webcam1_image_shape, webcam2_image_shape]  

    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}
    
    dataset = LeRobotDataset.create(
        repo_id="olingoudey/" + dataset_name,
        fps=t_cfg.fps,
        root=Path('./data/' + dataset_name + str(random.randint(0, 1000))), # random numbers for safety
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * 2,
        batch_encoding_size=16,
    )

    teleop.connect()
    robot.connect()
    
    try:
        robot = reset_bw_episode(robot, teleop)
        task = input("Task name?:")
        #task = "Put the cube in the bowl" # preset for ease
        input("[hit Enter]")
        with VideoEncodingManager(dataset):
            while True:
                logging.info("New episode starting... ^C when done or to stop.")
                try:
                    try:
                        teleop_loop(teleop, robot, t_cfg.fps, display_data=t_cfg.display_data, duration=t_cfg.teleop_time_s, cameras=cameras, dataset=dataset, task=task) # send IPwebcam to teleop loop 
              
                    except KeyboardInterrupt:
                        chat = input("Save episode? (hit Enter/y for yes, n for no)")
                        if chat == "" or strtobool(chat):
                            logging.info("Saving episode (out)")
                            dataset.save_episode()
                            logging.info("Saved episode (out)")
                            chat = input("Type task: (hit Enter/"" to skip)")
                            if not chat == "":
                                task = chat
                        else:
                            logging.info("Deleting episode (out)")
                            dataset.clear_episode_buffer()
                            logging.info("Deleted episode (out)")
                    input("\nReset robot? (^C to exit)")
                    robot = reset_bw_episode(robot, teleop)
                    input("[hit Enter]")
                    #new_task = input("\nNew task? (hit only Enter to use same task) (^C to exit)") # environment scenario updated manually
                    #if new_task:
                    #    task = new_task
                except Exception as e:
                    print("Robot connection error?:", e)
                    print("Reconnecting... (Catch me!)")
                    time.sleep(1)
                    robot.disconnect()
                    robot = make_robot_from_config(t_cfg.robot)
                    robot.connect()
                    input("Continue?")
                    continue
                
    except KeyboardInterrupt:
    
        logging.info("\nExiting episode loop.")
        try:
            chat = input("Save dataset?\n")
            if chat == "" or strtobool(chat):
                chat = input("Push to hub?\n")
                if chat == "" or strtobool(chat):
                    write_dataset_card(dataset.root / "README.md")
                    logging.info("Great. Pushing.")
                    dataset.push_to_hub()
                    logging.info("Pushed.")
                else:
                    logging.info("Great. Saved locally only.")    
            else:
                raise KeyboardInterrupt # goto V
        except KeyboardInterrupt:
            logging.info("\nDeleting dataset...")
            # Remove temporary files
            chat = input("Are you sure you want to delete?!")
            if chat == "" or strtobool(chat):
                dataset.clear_episode_buffer() 
                if dataset.root and os.path.exists(dataset.root):
                    shutil.rmtree(dataset.root)
                    logging.info(f"Deleted dataset at /{dataset.root}")       
        except KeyboardInterrupt:
            pass
        webcam1_reader.stop()
        webcam2_reader.stop()
        webcam1_cap.release()
        webcam2_cap.release()
        input("[hit Enter to catch me]\n")
        
            
        for t in range(60, 0, -1):
            print(f"\r\Dropping in...! {t/20:.1f}s", end="", flush=True)
            time.sleep(0.05)
        logging.info("\rBye!      ") 
        
        robot.bus.disable_torque()
        if t_cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def test_policy(policy_path="/home/olin/Robotics/Projects/LeRobot/lerobot/outputs/train/2025-09-06/14-21-15_smolvla/checkpoints/last/pretrained_model", camera_urls=["rtsp://192.168.0.159:8080/h264_ulaw.sdp", "rtsp://192.168.0.151:8080/h264_ulaw.sdp"], signal={"flag":"GO", "instruction":"Put the colored blocks in the cardboard box"}):
    """ Runs the SmolVLA policy at policy_path."""
    _init_rerun(session_name="smolvla")
    #policy_path = "lerobot/smolvla_base" # to test the base model (it's weird and ineffective)
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="normal",
        use_degrees=False,
    )
    robot = SO101Follower(robot_config)
    robot.connect()
    
    smolvla_policy = SmolVLAPolicy.from_pretrained(policy_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("Policy made.")
    webcam1_url = camera_urls[0]
    webcam2_url = camera_urls[1]
    webcam1_cap = cv2.VideoCapture(webcam1_url)
    webcam2_cap = cv2.VideoCapture(webcam2_url)
    webcam1_reader = IPWebcamReader(webcam1_cap)
    webcam2_reader = IPWebcamReader(webcam2_cap)
    webcam1_reader.start()
    webcam2_reader.start()
    print("Giving cameras time...")
    time.sleep(2)
    if not webcam1_cap.isOpened() or not webcam2_cap.isOpened():
        time.sleep(1)
        print("More time...")
        if not webcam1_cap.isOpened() or not webcam2_cap.isOpened():
            raise RuntimeError("Cannot open IP webcam")
    if not webcam1_cap.isOpened() or not webcam2_cap.isOpened():
        raise RuntimeError("Cannot open IP webcam") 
    while webcam1_reader.frame is None:
        print("\rWaiting... on camera1")
        time.sleep(0.01)
    while webcam2_reader.frame is None:
        print("\rWaiting... on camera2")
        time.sleep(0.01)
    logging.info("Cameras on.")
    input("[hit Enter]")
    robot = reset_bw_episode(robot, None)
    instruction = signal["instruction"]
    try:
        single_task=input(f"Instruction ([Enter] for {instruction}):\n")
        if single_task == "":
            single_task = "Put the colored blocks in the cardboard box"
        
        while True:
            try:
                while not signal["flag"] == "STOP":
                    print(signal)
                    start_loop_t = time.perf_counter()
                    robot.get_observation()
                    printable_state = {name: round(robot.present_pos[name.split(".pos")[0]], 6) for i, name in enumerate(robot.action_features)}
                    #print("STATE:  ", printable_state)
                    joints_deg = np.array([robot.present_pos[name.split(".pos")[0]] for name in robot.action_features], dtype=np.float32)
                    frame1 = webcam1_reader.frame
                    frame2 = webcam2_reader.frame
                    observation_frame = {
                        "observation.state": joints_deg,   # robot state
                        "observation.images.side": frame1,
                        "observation.images.up": frame2,
                    }
                    
                    action_values = predict_action(
                        observation_frame,
                        smolvla_policy,
                        device=device,
                        use_amp=(device.type == "cuda"),
                        task=single_task,
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
                robot = reset_bw_episode(robot, None)
                smolvla_policy.reset()
                input("[Enter]")
        print("Leaving model inference...")
        raise KeyboardInterrupt("Essentially a keyboard interrupt")
    except KeyboardInterrupt:
        webcam1_reader.stop()
        webcam2_reader.stop()
        webcam1_cap.release()
        webcam2_cap.release()
        input("[hit Enter to catch me]\n") 
        for t in range(60, 0, -1):
            print(f"\r\Dropping in...! {t/20:.1f}s", end="", flush=True)
            time.sleep(0.05)
        logging.info("\rBye!      ")
        robot.bus.disable_torque()
        robot.disconnect()

def reset_bw_episode(robot, teleop):
    print("Resetting position")
    try:
        robot.reset_position()
        print("robot reset...")
        
        
        if teleop:
            calculated_ee_pos = teleop.kinematics.forward_kinematics(np.array([robot.present_pos[name] for name in teleop.joint_names]))
            print("resetting teleop's target pose")
            teleop.reset(calculated_ee_pos)

            
            print("actuating to target pose at start of teleop loop")
        return robot
    except RuntimeError as e:
        print("Robot connection error?:", e)
        print("Reconnecting... (Catch me!)")
        time.sleep(1)
        robot.disconnect()
        robot = make_robot_from_config(teleop_config().robot)
        robot.connect()
        return robot

def test_webcam(url="https://192.168.0.159:8080/shot.jpg"):
    """ Used for testing the web cam at a given url. """
    import matplotlib.pyplot as plt
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Failed to open IP webcam")
        return
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to grab frame")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame_rgb)
    plt.axis("off")
    plt.show()
    input(".")

def dummy_dataset():
    """ Used for verifying the format of datasets. """
    robot_motors_ft = {f"{motor}.pos": float for motor in ["A", "B", "C", "D", "E", "F"]}
    webcam1_image_shape = probe_shape("https://192.168.0.151:8080/shot.jpg")
    robot_observation_features = {
        **robot_motors_ft,
        "side": webcam1_image_shape,
    }   # overriding property of so101

    action_features = hw_to_dataset_features(robot_motors_ft, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot_observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id="/datasets",
        fps=30,
        root=Path('./data' + str(random.randint(0, 100))),
        robot_type="normal",
        features=dataset_features,
        use_videos=True,
        image_writer_processes=2,
        image_writer_threads=2 * 1,
        batch_encoding_size=16,
    )
    test_record_loop(dataset)
    dataset.save_episode()

def teleoperate(cfg: TeleoperateConfig):
    """ Just teleoperate function """
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.reset_position()
    print("robot reset...")
    robot.get_observation()
    calculated_ee_pos = teleop.kinematics.forward_kinematics(np.array([robot.present_pos[name] for name in teleop.joint_names]))
    
    print("resetting teleop's target pose")
    teleop.reset(calculated_ee_pos)
    
    
    print("actuating to target pose")
    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s, cameras=(camera1, camera2))
    except KeyboardInterrupt:
        print("Exiting!")
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()

def probe_shape(url):
    """ Helper function to get the dimensions of images. No longer used """
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open IP webcam stream")
    print("Giving camera time...")
    time.sleep(1)
    ret, frame = cap.read()
    
    if not ret:
        raise RuntimeError("Failed to grab frame")
    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    print("Frame:", frame.shape, frame.dtype)
    cap.release()
    
    return frame.shape
 
def get_dataset(path, repo_id):
    """ Interpreter helper """
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=Path(path),
    )
    print(dataset)
    return dataset
 
def dataset_folder_task_info(dataset_dir): 
    datasets_root = Path(dataset_dir)
    tasks = set()
    # Iterate over all subfolders in the root
    for dataset_dir in datasets_root.iterdir():
        if not dataset_dir.is_dir():
            continue

        tasks_file = dataset_dir / "meta" / "tasks.jsonl"
        if not tasks_file.exists():
            print(f"\nNo tasks.jsonl found in {dataset_dir}")
            continue

        print(f"\nTasks in dataset: {dataset_dir.name}")
        with open(tasks_file, "r") as f:
            for line in f:
                task_entry = json.loads(line)
                task = task_entry["task"]
                tasks.add(task)
                print("  -", task)
    print("\n Combined unique tasks:", list(tasks))

def synonomize(output_dir, dataset_dir):
    from custom_brains.synonym_llm import Synonym_Agent
    poet = Synonym_Agent()
    dataset_root = Path(dataset_dir)
    out_dir = Path(output_dir)
    tasks_file = dataset_root / "meta" / "tasks.jsonl"
    episode_stats_file = dataset_root / "meta" / "episodes_stats.jsonl"
    with open(tasks_file, "r") as f:
        tasks = [json.loads(l)["task"] for l in f] # like enumerate()
            
    new_tasks = {}
    new_stats = []
    global_task_index = 0
    with open(episode_stats_file, "r") as f:
        for line in f:
            ep = json.loads(line)
            task = tasks[ep["stats"]["task_index"]["min"][0]]
            new_task_description = poet.synonomize(task)
            old_count = ep["stats"]["task_index"]["count"][0]
            new_index = global_task_index
            new_tasks[new_task_description] = new_index
            ep["stats"]["task_index"] = {"min": [new_index], "max": [new_index], "mean": [float(new_index)], "std": [float(new_index)], "count": [old_count]}
            global_task_index += 1
            new_stats.append(ep)
    
    with open(out_dir / "meta/episodes_stats.jsonl", "w") as f:
        for stat in new_stats:
            f.write(json.dumps(stat) + "\n")

    with open(out_dir / "meta/tasks.jsonl", "w") as f:
        for task in new_tasks.keys():
            task_entry = {"task_index": new_tasks[task], "task": task}
            f.write(json.dumps(task_entry) + "\n")
    
    # ---- Copy remaining files from meta/, data/, and videos/ ----
    skip = {
        "meta/episodes_stats.jsonl",
        "meta/tasks.jsonl",
    }

    for folder in ["meta", "data", "videos"]:
        src_dir = dataset_root / folder
        dst_dir = out_dir / folder
        dst_dir.mkdir(parents=True, exist_ok=True)

        for file in src_dir.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(dataset_root)
                if rel_path.as_posix() not in skip:
                    dst_file = dst_dir / file.relative_to(src_dir)
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file, dst_file)
                

def merge_datasets(out_dir, *dataset_dirs):
    """
    Used to combine two identically formatted datasets, likely because a recording session was interrupted. 
    Kind of a force. Don't use without editing.
    """
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    # merged metadata
    merged_episodes = []
    merged_stats = []
    merged_tasks = {}
    
    global_episode_index_episodes = 0
    global_episode_index_stats = 0
    video_up_index = 0
    video_side_index = 0
    global_episode_index_data = 0
    global_task_index = 0

    (up_dir := out_dir / "videos/chunk-000/observation.images.up").mkdir(parents=True, exist_ok=True)
    (side_dir := out_dir / "videos/chunk-000/observation.images.side").mkdir(parents=True, exist_ok=True)
    chunk_data_dir = out_dir / "data" / "chunk-000"
    chunk_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting merge of {len(dataset_dirs)} datasets into {out_dir}\n")

    for d_idx, d in enumerate(dataset_dirs):
        d = Path(d)
        print(f"Processing dataset {d_idx + 1}/{len(dataset_dirs)}: {d}")
        data_chunk = d / "data/chunk-000"
        video_chunk = d / "videos/chunk-000"
        
        tasks_file = d / "meta/tasks.jsonl"
        with open(tasks_file) as f:
            tasks = [json.loads(l) for l in f]
        print(f"  Found {len(tasks)} tasks in this dataset")

        for episode_file in sorted(data_chunk.glob("episode_*.parquet")):
            new_data_name = f"episode_{global_episode_index_data:06d}.parquet"
            shutil.copy(episode_file, out_dir / "data/chunk-000" / new_data_name)
            global_episode_index_data += 1

        src_up_dir = video_chunk / "observation.images.up"
        for video_file in sorted(src_up_dir.glob("episode_*.mp4")):
            new_up_name = f"episode_{video_up_index:06d}.mp4"
            print("  Copying video to up_dir at", up_dir / new_up_name)
            shutil.copy(video_file, up_dir / new_up_name)
            video_up_index += 1

        src_side_dir = video_chunk / "observation.images.side"
        for video_file in sorted(src_side_dir.glob("episode_*.mp4")):
            new_side_name = f"episode_{video_side_index:06d}.mp4"
            print("  Copying video to side_dir at", side_dir / new_side_name)
            shutil.copy(video_file, side_dir / new_side_name)
            video_side_index += 1

        task_index_local_to_global = {}
        with open(d / "meta" / "tasks.jsonl") as f:
            print("  Opening tasks")
            for line in f:
                task_entry = json.loads(line)
                task_index = task_entry["task_index"]
                task = task_entry["task"]
                if task in merged_tasks:
                    task_index_local_to_global[task_index] = merged_tasks[task]          
                else:
                    merged_tasks[task] = global_task_index
                    task_index_local_to_global[task_index] = global_task_index
                    global_task_index += 1
                
        with open(d / "meta/episodes.jsonl") as f:
            print("  Copying episodes.jsonl from", d)
            for line in f:
                ep = json.loads(line)
                ep["episode_index"] = global_episode_index_episodes
                merged_episodes.append(ep)
                global_episode_index_episodes += 1
                
        with open(d / "meta/episodes_stats.jsonl") as f:
            print("  Copying episodes_stats.jsonl from", d)
            for line in f:
                stat = json.loads(line)
                stat["episode_index"] = global_episode_index_stats
                merged_stats.append(stat)
                
                
                new_index = task_index_local_to_global[stat["stats"]["task_index"]["min"][0]] 
                old_count = stat["stats"]["task_index"]["count"][0]
                stat["stats"]["episode_index"] = {"min": [global_episode_index_stats], "max": [global_episode_index_stats], "mean": [float(global_episode_index_stats)], "std": [float(global_episode_index_stats)], "count": [old_count]}
                stat["stats"]["task_index"] = {"min": [new_index], "max": [new_index], "mean": [float(new_index)], "std": [float(new_index)], "count": [old_count]}
                global_episode_index_stats += 1
        # Cumulative totals after each dataset
        print(f"  Cumulative totals after dataset {d_idx + 1}:")
        print(f"    Total episodes merged: {global_episode_index_episodes}")
        print(f"    Total stats merged: {global_episode_index_stats}")
        print(f"    Total videos copied (up): {video_up_index}, (side): {video_side_index}")
        print(f"    Total unique tasks: {len(merged_tasks)}\n")

    # write merged metadata
    with open(out_dir / "meta/episodes.jsonl", "w") as f:
        for ep in merged_episodes:
            f.write(json.dumps(ep) + "\n")

    with open(out_dir / "meta/episodes_stats.jsonl", "w") as f:
        for stat in merged_stats:
            f.write(json.dumps(stat) + "\n")

    with open(out_dir / "meta/tasks.jsonl", "w") as f:
        for task in merged_tasks.keys():
            task_entry = {"task_index": merged_tasks[task], "task": task}
            f.write(json.dumps(task_entry) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "so101_follower",
        "total_episodes": len(merged_episodes),
        "total_frames": sum(ep["length"] for ep in merged_episodes),
        "total_tasks": len(merged_tasks),
        "total_videos": len(merged_episodes) * 2,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{len(merged_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", 
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    }

    first_info_path = Path(dataset_dirs[0]) / "meta" / "info.json"
    if first_info_path.exists():
        with open(first_info_path) as f:
            first_info = json.load(f)
            info["features"] = first_info.get("features", {})

    with open(out_dir / "meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"Merging complete. Total episodes: {len(merged_episodes)}, total tasks: {len(merged_tasks)}")


def check_episode_stats(file_path, eps=1e-4):
    """
    Scan episodes_stats.jsonl for anomalies in std of actions and states.
    eps: threshold below which std is considered suspiciously low.
    """
    file_path = Path(file_path)
    anomalies = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep_stat = json.loads(line)
            ep_idx = ep_stat["episode_index"]
            stats = ep_stat["stats"]

            for key in ["action", "observation.state"]:
                stat_std = stats.get(key, {}).get("std", None)
                if stat_std is None:
                    anomalies.append((ep_idx, key, "Missing std"))
                    continue

                # handle list of values
                if isinstance(stat_std[0], list):
                    std_values = np.array(stat_std).flatten()
                else:
                    std_values = np.array(stat_std)

                if np.any(std_values <= eps):
                    anomalies.append((ep_idx, key, std_values.tolist()))

    if anomalies:
        print(f"Found {len(anomalies)} anomalies:")
        for ep_idx, key, std_vals in anomalies:
            print(f"Episode {ep_idx}, field '{key}', std={std_vals}")
    else:
        print("No anomalies found.")
 
def teleop_config():
    """ Helper function to create a TeleoperateConfig """
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="normal",
        use_degrees=False,
    )

    follower = SO101Follower(robot_config)
    follower.connect()

    teleop_config = TeleoperateConfig(
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
    return teleop_config 

def write_dataset_card(filename="README.md"):
    content = """
---
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---
This dataset was created using [my fork of LeRobot](https://github.com/ogoudey/lerobot).

## Joint calibration
Joint calibration for the featured SO-101 is on [Github](https://github.com/ogoudey/lerobot-calibration)

## Dataset Structure

[meta/info.json](meta/info.json):
"""
    with open(filename, "w") as f:
        f.write(content)
    print(f"Wrote {filename} to current folder.")

### Dataset Wrangling
# d = test.get_dataset(path="data/h485", repo_id"olingoudey/put_the_stuffed_animal_in_the_bowl") # PRovide root (relative path) and remote repo
# test.write_dataset_card("data/h485/README.md")
# d.push_to_hub() # to remote repo
# d.pull_from_repo() # to local root, I think, since `root` is specified. If root folder doesn't exist, idk.

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
    record_dataset(dataset_name="blocks_box2", camera_refs=[2, 4]) 
    #teleoperate(teleop_config())
    
    #test_policy("/home/mulip-guest/LeRobot/lerobot/outputs/stationary_env_3k/pretrained_model", camera_urls=["rtsp://10.243.112.170:8080/h264_ulaw.sdp", "rtsp://10.243.63.69:8080/h264_ulaw.sdp"])
    

    
    
    #test_policy("/home/mulip-guest/LeRobot/lerobot/outputs/blocks_box/checkpoints/021000/pretrained_model", camera_urls=["rtsp://10.243.59.185:8080/h264_ulaw.sdp", "rtsp://10.243.126.188:8080/h264_ulaw.sdp"])

    

if __name__ == "__main__":
    #main_with_signal({"flag":"STOP", "instruction": "STOP")
    main()
