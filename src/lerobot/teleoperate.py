# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so100_follower \
  --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
  --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
  --robot.id=bimanual_follower \
  --robot.cameras='{
    left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
    top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
    right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
  }' \
  --teleop.type=bi_so100_leader \
  --teleop.left_arm_port=/dev/tty.usbmodem5A460828611 \
  --teleop.right_arm_port=/dev/tty.usbmodem5A460826981 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import rerun as rr
import numpy as np

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
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
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    
    print("Teleop class:", type(teleop).__name__)
    print("Beginning loop...")
    while True:
        loop_start = time.perf_counter()
        
        observation = robot.get_observation()
        action = teleop.get_action()
        
        if display_data:
            
            log_rerun_data(observation, action)
        
        present_pos = np.array([robot.present_pos[name] for name in teleop.joint_names])    # convert to np_array for kinematics
        print("Present:", robot.present_pos, "\n(The values go into K functions in degrees))
        ee_pos = teleop.kinematics.forward_kinematics(present_pos)
        translation = ee_pos[:3, 3]
        if type(teleop).__name__ == "KeyboardEndEffectorTeleop":
            """ Re-Calculate action """        
            deltas = [action["delta_x"], action["delta_y"], action["delta_z"]]
            translation += np.array(deltas)
            ee_pos[:3, 3] = translation

            np_pos = teleop.kinematics.inverse_kinematics(present_pos, ee_pos)
            action = {name + '.pos': float(val) for name, val in zip(teleop.joint_names, np_pos)} # convert back to action dict
        #print("Action:", action)
        robot.send_action(action)
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start
        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'VALUE':>7}")
        print(f"{'old_x':<{display_len}} | {ee_pos[:3, 3][0]:>7.2f}")
        print(f"{'old_y':<{display_len}} | {ee_pos[:3, 3][1]:>7.2f}")
        print(f"{'old_z':<{display_len}} | {ee_pos[:3, 3][2]:>7.2f}")
        print(f"{'dx':<{display_len}} | {deltas[0]:>7.2f}")
        print(f"{'dy':<{display_len}} | {deltas[1]:>7.2f}")
        print(f"{'dz':<{display_len}} | {deltas[2]:>7.2f}")
        print(f"{'x':<{display_len}} | {translation[0]:>7.2f}")
        print(f"{'y':<{display_len}} | {translation[1]:>7.2f}")
        print(f"{'z':<{display_len}} | {translation[2]:>7.2f}")
        for motor, value in action.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 5)


@draccus.wrap()
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
        robot.bus.disable_torque()
        robot.stop()
    except NameError:
        print("robot.bus.disable_torque()\nrobot.stop()\n\n...is not a thing.")
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    teleoperate()


if __name__ == "__main__":
    main()
