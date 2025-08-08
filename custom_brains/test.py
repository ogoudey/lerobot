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

import draccus
import rerun as rr



from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardJointTeleopConfig
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

    logger.info(f"Motors: {robot.bus.motors}")

    try:
        teleop_loop(teleop, robot, cfg.fps, display_data=cfg.display_data, duration=cfg.teleop_time_s)
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()



def main():
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="my_robot",
    )

    follower = SO101Follower(robot_config)
    follower.connect()

    teleop_config = TeleoperateConfig(
        robot = robot_config,
        teleop = KeyboardJointTeleopConfig(
            id="teleop1",
            calibration_dir=Path("."),
            mock=False,
        ),
        fps=30,
        teleop_time_s=60.0,
        display_data=False,
    )
    
    teleoperate(teleop_config)


if __name__ == "__main__":
    main()
