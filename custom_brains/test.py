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




from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardJointTeleopConfig, KeyboardEndEffectorTeleopConfig
from pathlib import Path
from lerobot.teleoperate import teleop_loop

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
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop = make_teleoperator_from_config(t_cfg.teleop)
    robot = make_robot_from_config(t_cfg.robot)

    teleop.connect()
    robot.connect()
    
    #
    # connect to IPwebcam
    #
    
    try:
        robot.reset_position()
        input("Environment set up?") # environment scenario updated manually
        while True:
            print("New episode starting...")
            episode_data = {}
            done = False
            try:
                teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s, video_stream=None, episode_data=episode_data) # send IPwebcam to teleop loop
            
      
            except KeyboardInterrupt:
                print("Ending episode.")
                if strtobool(input("Save episode?"))
                    # validate last step (?)
                    print("Saving episode...")
                    # Form Lerobot episode from episode_data
                    
                    print("Episode saved.")
                else:
                    print("Ignoring episode.")
            robot.reset_position() # use default start position
            input("Environment set up? (^C to exit)") # environment scenario updated manually
    except KeyboardInterrupt:
        if strtobool(input("Save dataset?"))
            print("Saving dataset.")
            # Finish writing dataset
        else:
            print("Deleting dataset.")
            # Remove temporary files
  
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


def main():
    
    
    teleoperate(teleop_config())


if __name__ == "__main__":
    main()
