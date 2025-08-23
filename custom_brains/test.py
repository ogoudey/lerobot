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

def record_dataset():
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
            repo_id="/olindatasets",
            fps=t_cfg.fps,
            #root=Path('./data' + str(random.randint(0, 100))),
            root=Path('./dataset2'),
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * 2, # 2 times cameras
            batch_encoding_size=16,
        )
    except FileExistsError:
        dataset = LeRobotDataset('dataset')

    teleop.connect()
    robot.connect()
    
    try:
        robot.reset_position()
        input("\nEnvironment set up?\n") # environment scenario updated manually
        
        with VideoEncodingManager(dataset):
            while True:
                logging.info("New episode starting... ^C when done or to stop.")
                try:
                    teleop_loop(teleop, robot, t_cfg.fps, display_data=t_cfg.display_data, duration=t_cfg.teleop_time_s, video_streams=urls, dataset=dataset) # send IPwebcam to teleop loop 
          
                except KeyboardInterrupt:
                    logging.info("Saving episode (out)")
                    dataset.save_episode()
                    logging.info("Saved episode (out)")
                input("\nReset robot? (^C to exit)")
                robot.reset_position() # use default start position
                input("\nEnvironment set up? (^C to exit)") # environment scenario updated manually
    except KeyboardInterrupt:
    
        logging.info("\nExiting episode loop.")
        try:
            chat = input("Save dataset?\n")
            if strtobool(chat) or chat == "":
                logging.info("Great. Saved.")
            else:
                raise KeyboardInterrupt # goto V
        except KeyboardInterrupt:
            logging.info("\nDeleting dataset...")
            # Remove temporary files
            if strtobool(input("Are you sure you want to delete?!")):
                dataset.clear_episode_buffer() 
                if dataset.root and os.path.exists(dataset.root):
                    shutil.rmtree(dataset.root)
                    logging.info(f"Deleted dataset at /{dataset.root}")       
        except KeyboardInterrupt:
            pass
        # Safely quit
        input("[hit Enter to catch me]\n")
        
            
        for t in range(60, 0, -1):
            logging.info(f"\r\Dropping in...! {t/20:.1f}s", end="", flush=True)
            time.sleep(0.05)
        logging.info("\rBye!      ") 
        
        robot.bus.disable_torque()
        if t_cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()

def test_policy():
    """ Runs the SmolVLA policy at policy_path. Finicky, not working fully. """
    
    policy_path = "/home/olin/Robotics/Projects/LeRobot-Projects/lerobot/outputs/train/2025-08-18/13-40-25_smolvla/checkpoints/last/pretrained_model"
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="my_robot",
        use_degrees=False,
    )
    robot = SO101Follower(robot_config)
    robot.connect()
    
    smolvla_policy = SmolVLAPolicy.from_pretrained()
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
    if not webcam1_cap.isOpened() or not webcam2_cap.isOpened():
        raise RuntimeError("Cannot open IP webcam") 
    while webcam1_reader.frame is None:
        time.sleep(0.01)
    while webcam2_reader.frame is None:
        time.sleep(0.01)
    logging.info("Cameras on.")
    robot.reset_position()
    try:
        single_task="Pick up the object"
        while True:
            try:
                while True:
                    start_loop_t = time.perf_counter()
                    joint_state = robot.get_observation()
                    joints_deg = np.array([robot.present_pos[name.split(".pos")[0]] for name in robot.action_features])
                    camera_1 = webcam1_reader.frame
                    camera_2 = webcam2_reader.frame
                    observation_frame = {
                        "observation.state": np.array(joints_deg, dtype=np.float32),   # robot state
                        "observation.images.front": webcam1_reader.frame,
                        "observation.images.side": webcam2_reader.frame
                    }
                    action_values = predict_action(
                        observation_frame,
                        smolvla_policy,
                        device=device,
                        use_amp=(device.type == "cuda"),
                        task=single_task,
                        robot_type=robot.robot_type,
                    )
                    action = {key: action_values[i].item() for i, key in enumerate(robot.action_features)}
                    logging.info("Sending action")
                    robot.send_action(action)
                    dt_s = time.perf_counter() - start_loop_t
                    busy_wait(1 / 15 - dt_s) # this is 1 / fps - dt_s
            except KeyboardInterrupt:
                single_task=input("New task? (^C to Exit)\n")   # whether this changes anything is unclear
    except KeyboardInterrupt:
        input("[hit Enter to catch me]\n") 
        for t in range(60, 0, -1):
            logging.info(f"\r\Dropping in...! {t/20:.1f}s", end="", flush=True)
            time.sleep(0.05)
        logging.info("\rBye!      ") 
        robot.bus.disable_torque()
        robot.disconnect()

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
    robot.connect()

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

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab frame")
    frame = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    print("Frame:", frame.shape, frame.dtype)
    cap.release()
    
    return frame.shape
  
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

    #record_dataset()
    
    test_policy()

if __name__ == "__main__":
    main()
