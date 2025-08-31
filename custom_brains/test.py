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

# --- Third-party ---
import cv2
import numpy as np
import rerun as rr
import torch

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

# --- LeRobot: teleoperate ---
from lerobot.teleoperate import (
    TeleoperateConfig,
    CameraReader,
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

logger = logging.getLogger(__name__)

def record_dataset(dataset_name="dataset3"):
    t_cfg = teleop_config()

    init_logging()
    logging.info(pformat(asdict(t_cfg)))
    if t_cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(t_cfg.teleop)
    robot = make_robot_from_config(t_cfg.robot)
    input("Start cameras... [hit Enter to continue]")
    webcam1_url = "rtsp://192.168.0.159:8080/h264_ulaw.sdp"
    webcam2_url = "rtsp://192.168.0.151:8080/h264_ulaw.sdp"
    urls = [webcam1_url, webcam2_url]
    try:
        webcam1_image_shape = probe_shape(webcam1_url)
        webcam2_image_shape = probe_shape(webcam2_url)
        #laptop_image_shape = probe_shape(0)
    except RuntimeError as e:
        logging.info("Cannot connect to webcams...")
        raise RuntimeError
        webcam1_image_shape, webcam2_image_shape = (0), (0)
    robot_observation_features = {
        **robot._motors_ft,
        "front": webcam1_image_shape,
        "side": webcam2_image_shape,
        #"observation/images/front": laptop_image_shape,
    }   # overriding property of so101
    
    

    action_features = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot_observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}

    try:
        dataset = LeRobotDataset.create(
            repo_id="olindatasets",
            fps=t_cfg.fps,
            #root=Path('./data' + str(random.randint(0, 100))),
            root=Path('./data/' + dataset_name + str(random.randint(0, 1000))),
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * 2, # 2 times cameras
            batch_encoding_size=16,
        )
    except FileExistsError:
        root = Path('./data') / dataset_name
        dataset = LeRobotDataset(repo_id="/olindatasets", root=root)

    teleop.connect()
    robot.connect()
    
    try:
        robot = reset_bw_episode(robot, teleop)
        task = "Put the cube in the bowl"
        input("[hit Enter]")
        #task = input("\nWhat's the task name?\n")   # environment set up manually
        with VideoEncodingManager(dataset):
            while True:
                logging.info("New episode starting... ^C when done or to stop.")
                try:
                    try:
                        teleop_loop(teleop, robot, t_cfg.fps, display_data=t_cfg.display_data, duration=t_cfg.teleop_time_s, video_streams=urls, dataset=dataset, task=task) # send IPwebcam to teleop loop 
              
                    except KeyboardInterrupt:
                        chat = input("Save episode? (hit Enter/y for yes, n for no)")
                        if chat == "" or strtobool(chat):
                            logging.info("Saving episode (out)")
                            dataset.save_episode()
                            logging.info("Saved episode (out)")
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
                except Exception:
                    print("Robot connection error?:", e)
                    print("Reconnecting... (Catch me!)")
                    time.sleep(1)
                    robot.disconnect()
                    robot = make_robot_from_config(t_cfg.robot)
                    robot.connect()
                    continue
                
    except KeyboardInterrupt:
    
        logging.info("\nExiting episode loop.")
        try:
            chat = input("Save dataset?\n")
            if chat == "" or strtobool(chat):
                logging.info("Great. Saved.")
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
        # Safely quit
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

def test_policy():
    """ Runs the SmolVLA policy at policy_path. Finicky, not working fully. """
    
    policy_path = "/home/olin/Robotics/Projects/LeRobot/lerobot/outputs/train/2025-08-29/10-17-18_smolvla/checkpoints/last/pretrained_model"
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="my_robot",
        use_degrees=False,
    )
    robot = SO101Follower(robot_config)
    robot.connect()
    
    smolvla_policy = SmolVLAPolicy.from_pretrained(policy_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info("Policy made.")
    webcam1_url = "rtsp://192.168.0.159:8080/h264_ulaw.sdp"
    webcam2_url = "rtsp://192.168.0.151:8080/h264_ulaw.sdp"
    webcam1_cap = cv2.VideoCapture(webcam1_url)
    webcam2_cap = cv2.VideoCapture(webcam2_url)
    webcam1_reader = CameraReader(webcam1_cap)
    webcam2_reader = CameraReader(webcam2_cap)
    webcam1_reader.start()
    webcam2_reader.start()
    time.sleep(1)
    if not webcam1_cap.isOpened() or not webcam2_cap.isOpened():
        raise RuntimeError("Cannot open IP webcam") 
    while webcam1_reader.frame is None:
        time.sleep(0.01)
    while webcam2_reader.frame is None:
        time.sleep(0.01)
    logging.info("Cameras on.")
    robot = reset_bw_episode(robot, None)
    try:
        single_task=input("Instruction:\n")
        while True:
            try:
                while True:
                    start_loop_t = time.perf_counter()
                    robot.get_observation()
                    printable_state = {name: round(robot.present_pos[name.split(".pos")[0]], 3) for i, name in enumerate(robot.action_features)}
                    print("STATE:  ", printable_state)
                    joints_deg = np.array([robot.present_pos[name.split(".pos")[0]] for name in robot.action_features])
                    camera_1 = np.transpose(webcam1_reader.frame, (2, 0, 1))
                    camera_2 = np.transpose(webcam2_reader.frame, (2, 0, 1))
                    camera_1 = webcam1_reader.frame
                    camera_2 = webcam2_reader.frame
                    observation_frame = {
                        "observation.state": np.array(joints_deg, dtype=np.float32),   # robot state
                        "observation.images.front": camera_1,
                        "observation.images.side": camera_2,
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
                    printable_action = {key: round(action_values[i].item(), 3) for i, key in enumerate(robot.action_features)}
                    print("ACTION: ", printable_action)
                    robot.send_action(action)
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / 4 - dt_s) # this is 1 / fps - dt_s
            except KeyboardInterrupt:
                single_task=input("New task? (^C to Exit)\n")   # whether this changes anything is unclear
                robot = reset_bw_episode(robot, None)
    except KeyboardInterrupt:
        input("[hit Enter to catch me]\n") 
        for t in range(60, 0, -1):
            print(f"\r\Dropping in...! {t/20:.1f}s", end="", flush=True)
            time.sleep(0.05)
        logging.info("\rBye!      ") 
        robot.bus.disable_torque()
        robot.disconnect()

def reset_bw_episode(robot, teleop):
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

    # Convert BGR → RGB
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
        #"observation/images/front": laptop_image_shape,
    }   # overriding property of so101

    action_features = hw_to_dataset_features(robot_motors_ft, "action", use_video=True)
    obs_features = hw_to_dataset_features(robot_observation_features, "observation", use_video=True)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id="/datasets",
        fps=30,
        root=Path('./data' + str(random.randint(0, 100))),
        robot_type="my_robot",
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
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        print("Exiting!")
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()

def probe_shape(url):
    """ Helper function to get the dimensions of images """
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
 
import json
import shutil
from pathlib import Path

def merge_datasets(out_dir, *dataset_dirs):
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    # merged metadata
    merged_episodes = []
    merged_stats = []
    merged_tasks = [{"task_index": 0, "task": "Put the cube in the bowl"}]
    global_episode_index_episodes = 0
    global_episode_index_stats = 0
    video_front_index = 0
    video_side_index = 0
    global_episode_index_data = 0

    (front_dir := out_dir / "videos/chunk-000/observation.images.front").mkdir(parents=True, exist_ok=True)
    (side_dir := out_dir / "videos/chunk-000/observation.images.side").mkdir(parents=True, exist_ok=True)
    chunk_data_dir = out_dir / "data" / "chunk-000"
    chunk_data_dir.mkdir(parents=True, exist_ok=True)

    for d in dataset_dirs:
        d = Path(d)
        data_chunk = d / "data/chunk-000"
        video_chunk = d / "videos/chunk-000"
        
        for episode_file in sorted(data_chunk.glob("episode_*.parquet")):
            new_data_name = f"episode_{global_episode_index_data:06d}.parquet"
            shutil.copy(episode_file, out_dir / "data/chunk-000" / new_data_name)
            global_episode_index_data += 1

        src_front_dir = video_chunk / "observation.images.front"
        for video_file in sorted(src_front_dir.glob("episode_*.mp4")):
            new_front_name = f"episode_{video_front_index:06d}.mp4"
            shutil.copy(video_file, front_dir / new_front_name)
            video_front_index += 1

        src_side_dir = video_chunk / "observation.images.side"
        for video_file in sorted(src_side_dir.glob("episode_*.mp4")):
            new_side_name = f"episode_{video_side_index:06d}.mp4"
            shutil.copy(video_file, side_dir / new_side_name)
            video_side_index += 1

        # load and update metadata
        with open(d / "meta" / "episodes.jsonl") as f:
            print("Copying data from", d)
            for line in f:
                ep = json.loads(line)
                ep["episode_index"] = global_episode_index_episodes
                merged_episodes.append(ep)
                global_episode_index_episodes += 1

        with open(d / "meta" / "episodes_stats.jsonl") as f:
            for line in f:
                stat = json.loads(line)
                stat["episode_index"] = global_episode_index_stats
                merged_stats.append(stat)
                global_episode_index_stats += 1

    # write merged metadata
    with open(out_dir / "meta/episodes.jsonl", "w") as f:
        for ep in merged_episodes:
            f.write(json.dumps(ep) + "\n")

    with open(out_dir / "meta/episodes_stats.jsonl", "w") as f:
        for stat in merged_stats:
            f.write(json.dumps(stat) + "\n")

    with open(out_dir / "meta/tasks.jsonl", "w") as f:
        for task in merged_tasks:
            f.write(json.dumps(task) + "\n")

    # generate merged info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type": "so101_follower",
        "total_episodes": len(merged_episodes),
        "total_frames": sum(ep["length"] for ep in merged_episodes),
        "total_tasks": len(merged_tasks),
        "total_videos": len(merged_episodes) * 2,  # adjust if you have more/less videos per episode
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
        id="my_robot",
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
        
def main():
    """ A repetoire of useful main functions: """
    #test_webcam("https://192.168.0.159:8080/shot.jpg")
    #test_webcam("https://192.168.0.151:8080/shot.jpg")
    
    #dummy_dataset()

    merge_datasets("data/merged", "data/e752",  "data/e265")
    #check_episode_stats("data/merged/meta/episodes_stats.jsonl")
    #record_dataset("e")
    #teleoperate(teleop_config())
    #test_policy()

if __name__ == "__main__":
    main()
