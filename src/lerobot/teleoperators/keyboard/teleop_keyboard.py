#!/usr/bin/env python

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

import logging
import os
import sys
import time
import threading
from queue import Queue, Empty
from typing import Any
from pathlib import Path
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
import numpy as np
from ..teleoperator import Teleoperator
from .configuration_keyboard import KeyboardEndEffectorTeleopConfig, KeyboardTeleopConfig, KeyboardJointTeleopConfig
import socket

AS_TELEOP_SERVER = True
PYNPUT_AVAILABLE = False
try:
    if ("DISPLAY" not in os.environ) and ("linux" in sys.platform):
        logging.info("No DISPLAY set. Skipping pynput import.")
        raise ImportError("pynput blocked intentionally due to no display.")

    from pynput import keyboard
except ImportError:
    keyboard = None
    PYNPUT_AVAILABLE = False
except Exception as e:
    keyboard = None
    PYNPUT_AVAILABLE = False
    logging.info(f"Could not import pynput: {e}")


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type

        self.event_queue = Queue()
        self.current_pressed = {}
        self.listener = None
        self.logs = {}
        self.urdf_path = Path("/home/olin/Robotics/Projects/LeRobot/lerobot/custom_brains/so101_new_calib.urdf").as_posix() # change
        
        max_joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ] # for reference
        
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        
        print(f"Loading URDF from: {self.urdf_path} (is file? {os.path.isfile(self.urdf_path)})")
        self.kinematics = RobotKinematics(self.urdf_path, 'gripper_frame_link', self.joint_names)
        
        # Checking order of joints so solver is aligned #
        kinematics_joint_order = list(self.kinematics.robot.model.names)[2:]
        assert kinematics_joint_order == self.joint_names
        assert self.kinematics.joint_names == self.joint_names    

        self.signal = {} # gets from vla_complex
        
        self.teleop_port = 5008
        self.listening = False
        self.send_q = Queue()
        self._is_connected = False
    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        pass

    def run_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", self.teleop_port))
        server.listen()
        self.listening = True
        print("Teleop server waiting...")
        while self.listening:
            client_sock, addr = server.accept()
            print(f"Teleop client connected on {addr}")
            threading.Thread(
                target=self.handle_client,
                args=(client_sock,),
                daemon=True
            ).start()

    def handle_client(self, sock):
        stop_event = threading.Event()
        threading.Thread(
            target=self.recv_loop,
            args=(sock, stop_event),
            daemon=True
        ).start()
        threading.Thread(
            target=self.send_loop,
            args=(sock, self.send_q, stop_event),
            daemon=True
        ).start()
    
    def send_loop(self, sock: socket.socket, send_q: Queue, stop_event):
        print("Send loop started")
        try:
            while not stop_event.is_set():
                msg = send_q.get()
                sock.sendall((msg + "\n").encode())
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            stop_event.set()
            print("send_loop exiting")

    def recv_loop(self, sock: socket.socket, stop_event):
        print("Recv loop started")
        try:
            while not stop_event.is_set():
                
                key, is_pressed = self.recv_code(sock)
                if is_pressed:
                    if key == 'y':
                        self.signal["DECISION"] = "y"
                        print(self.signal)
                    elif key == 'n':
                        self.signal["DECISION"] = "n"

                self.event_queue.put((key, is_pressed))
        except (ConnectionResetError, OSError) as err:
            print(err)
        finally:
            stop_event.set()
            print("Disconnected recv_loop")
    
    def recv_code(self, sock: socket.socket) -> tuple:
        try:
            data = sock.recv(2)
            char = data[0:1].decode()
            is_pressed = bool(int(data[1:2].decode()))
            #print(f"Code: {(char, is_pressed)}")
            if not data:
                return None, None
            return char, is_pressed
        except OSError:
            return None, None
        
    def clear_queue(self):
        try:
            while True:
                self.event_queue.get_nowait()
        except Empty:
            pass

    def connect(self, signal) -> None:
        self.signal = signal
        self.clear_queue()
        if self.is_connected:
            print(f"Is listener alive? {self.listener.is_alive()}")

            return
            raise DeviceAlreadyConnectedError(
                "Keyboard is already connected. Do not run `robot.connect()` twice."
            )
        if AS_TELEOP_SERVER:
            self.listener = threading.Thread(target=self.run_server, daemon=True)
            self.listener.start()
        elif PYNPUT_AVAILABLE:
            logging.info("pynput is available - enabling local keyboard listener.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
                suppress=False
            )
            self.listener.start()
            self.listener.wait()
            print("Listener alive?", self.listener.is_alive()) 
        else:
            logging.info("pynput nor teleop_server available - skipping local keyboard listener.")
            self.listener = None
        self._is_connected = True

    def send_message(self, msg):
        try:
            if AS_TELEOP_SERVER:
                print(msg)
                self.send_q.put(msg)
            else:
                print(f">>> {msg}")
        except Exception as e:
            print(f"Error in send_message: {e}")

    def calibrate(self) -> None:
        pass

    def _on_press(self, key):
        print(f"{key} pressed!")
        if hasattr(key, "char") and key.char is not None:
            self.event_queue.put((key.char, True))
        else:
            self.event_queue.put((str(key), True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            self.event_queue.put((key.char, False))
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, disconnecting.")
            self.disconnect()

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            self.current_pressed[key_char] = is_pressed

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        print("[DEBUG] get_action() called")
        before_read_t = time.perf_counter()

        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()
        
        # Generate action based on current key states
        action = {key for key, val in self.current_pressed.items() if val}
        pressed = {key for key, val in self.current_pressed.items() if val}
        #print("[DEBUG] Action:", list(action))
        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return dict.fromkeys(action, None)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `robot.connect()` before `disconnect()`."
            )
        if self.listener is not None:
            self.listener.stop()



class KeyboardJointTeleop(KeyboardTeleop):
    """
    CustomTeleopClass
    """

    config_class = KeyboardJointTeleopConfig
    name = "keyboard_j"
    
    
    
    def __init__(self, config: KeyboardJointTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()
        
        self.joint_targets = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        
        # Key mapping: key -> (joint_name, direction)
        self.key_to_joint = {
            "q": ("shoulder_pan", +1),
            "a": ("shoulder_pan", -1),
            "w": ("shoulder_lift", +1),
            "s": ("shoulder_lift", -1),
            "e": ("elbow_flex", +1),
            "d": ("elbow_flex", -1),
            "r": ("wrist_flex", +1),
            "f": ("wrist_flex", -1),
            "t": ("wrist_roll", +1),
            "g": ("wrist_roll", -1),
            "y": ("gripper", +1),
            "h": ("gripper", -1),
        }
        
        self.step = 1
        
    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": list(self.arm.motors),
        }

    def _on_press(self, key):
        print(key, "pressed")
        if hasattr(key, "char"):
            key = key.char
            
        self.event_queue.put((key, True))

    def _on_release(self, key):
        print(key, "released")
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        self._drain_pressed_keys()

        # Generate action based on current key states
        for key, pressed in self.current_pressed.items():
            if pressed and key in self.key_to_joint:
                print("Key", key, "pressed.")
                joint, direction = self.key_to_joint[key]
                self.joint_targets[joint] += direction * self.step
            elif pressed:
                self.misc_keys_queue.put(key)
        
        self.current_pressed.clear()

        action_dict = {f"{joint}.pos": pos for joint, pos in self.joint_targets.items()}
        return action_dict
        
from ...model.kinematics import RobotKinematics    
        
        
class KeyboardEndEffectorTeleop(KeyboardTeleop):
    

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig):
        super().__init__(config)
        self.config = config
        self.misc_keys_queue = Queue()
        
        self.key_to_delta = {
            "i": ("x", +1),
            "k": ("x", -1),
            "j": ("y", +2),
            "l": ("y", -2),
            "u": ("z", +1),
            "o": ("z", -1),
        }
        
        self.key_to_orient = {
            "w": ("pitch", -1),
            "s": ("pitch", +1),
            "q": ("roll", -1),
            "e": ("roll", +1),
        }
        
        self.key_gripper = {
            "z": ("gripper", +1),
            "x": ("gripper", -1),
        }
        
        self.target_pos = {
            "x": 0.2,
            "y": 0,
            "z": 0.2,
            "roll": 0.0,
            "pitch": 90.0,
            "gripper": 0.0,
        }
        
        self.factor = 0.0015
        self.roll_pitch_factor = 0.9
        self.gripper_factor = 1
        
        

    @property
    def old_action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def action_features(self) -> dict:
        return {
            "dtype": "float32",
            "shape": (len(self.arm),),
            "names": {"motors": list(self.arm.motors)}, #
        }
    def reset(self, ee_pos):
        init_fk = ee_pos[:3, 3]
        print("3D pose:", init_fk)
        self.target_pos = {
            "x": init_fk[0],
            "y": init_fk[1],
            "z": init_fk[2],
            "roll": 0.0,
            "pitch": 90.0,
            "gripper": 0.0,
        }

    def rot_y(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[ c, 0, s],
                        [ 0, 1, 0],
                        [-s, 0, c]])

    def rot_z(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s, 0],
                        [s, c, 0],
                        [0, 0, 1]]),

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "KeyboardTeleop is not connected. You need to run `connect()` before `get_action()`."
            )
        self._drain_pressed_keys()
        # Generate action based on current key states
        for key, val in self.current_pressed.items():
            if not val:
                continue
            if val and key in self.key_to_delta:
                axis, direction = self.key_to_delta[key]
                self.target_pos[axis] += direction * self.factor
                
                # e.g. self.target_pos["roll"] += 1 * 0.005
            if val and key in self.key_to_orient:
                axis, direction = self.key_to_orient[key]
                self.target_pos[axis] += direction * self.roll_pitch_factor    
            if val and key in self.key_gripper:
                gripper, direction = self.key_gripper[key]
                adjust = direction * self.gripper_factor
                self.target_pos[gripper] += adjust
        return self.target_pos

    def _on_press(self, key):
        if hasattr(key, "char"):
            key = key.char
            if key == 'y':
                self.signal["DECISION"] = "y"
            elif key == 'n':
                self.signal["DECISION"] = "n"
        self.event_queue.put((key, True))

    def _on_release(self, key):
        if hasattr(key, "char"):
            key = key.char
        self.event_queue.put((key, False))

    


        
    

