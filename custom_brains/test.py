from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,

)
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    gamepad,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)

from lerobot.teleoperate import TeleoperateConfig

from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from distutils.util import strtobool

import draccus
import rerun as rr

import cv2


from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardJointTeleopConfig, KeyboardEndEffectorTeleopConfig
from pathlib import Path
from lerobot.teleoperate import teleop_loop

from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager

logger = logging.getLogger(__name__)

def teleoperate(cfg: TeleoperateConfig):
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

def record_dataset():
    t_cfg = teleop_config()
    
    init_logging()
    logging.info(pformat(asdict(t_cfg)))
    if t_cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(t_cfg.teleop)
    robot = make_robot_from_config(t_cfg.robot)
    webcam_url = "https://192.168.0.159:8080/shot.jpg"
    webcam_image_shape = probe_shape(webcam_url)
    laptop_image_shape = probe_shape(0)
    robot.observation_features = {
        **robot._motors_ft,
        # Inject your own cameras with fixed sizes
        "observation/images/front": (480, 640, 3),
        "observation/images/side": webcam_image_shape,
    }   # overriding property of so101
    
    

    action_features = hw_to_dataset_features(robot.action_features, "action", use_videos=True)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_videos=True)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=t_cfg.dataset.repo_id,
        fps=t_cfg.fps,
        root=t_cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=t_cfg.dataset.num_image_writer_processes,
        image_writer_threads=t_cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=t_cfg.dataset.video_encoding_batch_size,
    )

    teleop.connect()
    robot.connect()
    
    #
    # connect to IPwebcam
    #
    
    try:
        robot.reset_position()
        input("Environment set up?") # environment scenario updated manually
        
        with VideoEncodingManager(dataset):
            while True:
                print("New episode starting...")
                try:
                    teleop_loop(teleop, robot, t_cfg.fps, display_data=t_cfg.display_data, duration=t_cfg.teleop_time_s, video_stream=webcam_url, dataset=dataset) # send IPwebcam to teleop loop 
          
                except KeyboardInterrupt:
                    print("Ending episode.")
                    if strtobool(input("Save episode?")):
                        # validate last step (?)
                        print("Saving episode...")
                        # Form Lerobot episode from episode_data
                        
                        print("Episode saved.")
                    else:
                        print("Ignoring episode.")
                robot.reset_position() # use default start position
                input("Environment set up? (^C to exit)") # environment scenario updated manually
    except KeyboardInterrupt:
        try:
            if strtobool(input("Save dataset?")):
                print("Saving dataset.")
                dataset.save_episode()
                # Finish writing dataset
            else:
                print("Deleting dataset.")
                dataset.clear_episode_buffer()
        except KeyboardInterrupt:
            print("Deleting dataset and exiting (catch me!!).")
            # Remove temporary files
            
            # Safely quit
            robot.bus.disable_torque()
            robot.stop()
            if t_cfg.display_data:
                rr.rerun_shutdown()
            teleop.disconnect()
            robot.disconnect()

def probe_shape(url):
    cap = cv2.VideoCapture(ip_url)

    if not cap.isOpened():
        raise RuntimeError("Cannot open IP webcam stream")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to grab frame")

    print("Frame shape:", frame.shape)  # (height, width, channels)

    cap.release()
    return frame.shape
  
def teleop_config():
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


def test_webcam(url="https://192.168.0.159:8080/shot.jpg"):
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

    # Convert BGR → RGB if you like, for display or saving
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    plt.imshow(frame_rgb)
    plt.axis("off")
    plt.show()
    input(".")

def main():
    
    test_webcam(0)
    record_dataset()


if __name__ == "__main__":
    main()
