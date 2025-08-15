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

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def teleop_loop(
    teleop: Teleoperator, robot: Robot, fps: int, display_data: bool = False, duration: float | None = None, video_stream = None, episode_data={}, verbose=False
):
    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    
    position_weight, orientation_weight = 1.0, 0.1
    
    """ Calculate FK once for initial position """
    observation = robot.get_observation() # set robot.present_pos
    initial_joints_deg = np.array([robot.present_pos[name] for name in teleop.joint_names])    # convert to np_array for kinematics
    
    # Check kinematics
    kinematics_joint_order = list(teleop.kinematics.robot.model.names)[2:]
    assert kinematics_joint_order == teleop.joint_names
        
    calculated_ee_pos = teleop.kinematics.forward_kinematics(initial_joints_deg)
    
    init_fk = calculated_ee_pos[:3, 3]
    if type(teleop).__name__ == "KeyboardEndEffectorTeleop":
        teleop.target_pos["x"] =  init_fk[0]
        teleop.target_pos["y"] =  init_fk[1]    # overriding
        teleop.target_pos["z"] =  init_fk[2]

    teleop.kinematics.robot.update_kinematics()
    while True:
        loop_start = time.perf_counter()
        
        observation = robot.get_observation()
        
        #
        # get video stream frame
        #
        
        action = teleop.get_action()
        
        if display_data:
            log_rerun_data(observation, action)
        
        if type(teleop).__name__ == "KeyboardEndEffectorTeleop":
            """ Re-Calculate action """
            target_ee_pos = np.array([action["x"], action["y"], action["z"]])
            calculated_ee_pos[:3, 3] = target_ee_pos
            # Now affect R
            if True:
                target_pitch = np.deg2rad(action["pitch"])   # in degrees
                target_roll = np.deg2rad(action["roll"])
                R_new = rot_y(target_pitch) @ rot_z(target_roll)

                calculated_ee_pos[:3, :3] = R_new
            
            calculated_new_joints_deg = teleop.kinematics.inverse_kinematics(initial_joints_deg, calculated_ee_pos, position_weight, orientation_weight)
            target_gripper = action["gripper"]
            action = {name + '.pos': float(val) for name, val in zip(teleop.joint_names, calculated_new_joints_deg)} # convert back to action dict
            action["gripper.pos"] = target_gripper
        robot.send_action(action) # comment for mock?
        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start
        if verbose:
            print("\n" + "-" * (display_len + 10))

            for motor, value in action.items():
                print(f"{motor:<{display_len}} | {value:>7.2f}")

            print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action) + 10)


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
