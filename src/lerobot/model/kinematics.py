# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np


class RobotKinematics:
    """Robot kinematics using placo library for forward and inverse kinematics."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] = None,
    ):
        """
        Initialize placo-based kinematics solver.

        Args:
            urdf_path: Path to the robot URDF file
            target_frame_name: Name of the end-effector frame in the URDF
            joint_names: List of joint names to use for the kinematics solver
        """
        try:
            import placo
        except ImportError as e:
            print(e)
            raise ImportError(
                "placo is required for RobotKinematics. "
                "Please install the optional dependencies of `kinematics` in the package."
            ) from e
        
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        
        
        
        self.solver.mask_fbase(True)  # Fix the base
        #self.solver.enable_velocity_limits(True) # Added by Olin
        #self.solver.dt = 0.01 # "
        self.target_frame_name = target_frame_name

        # Set joint names
        self.joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names

        # Initialize frame task for IK
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))
        
        self.robot.update_kinematics()


    def forward_kinematics(self, joint_pos_deg, verbose=False):
        """
        Compute forward kinematics for given joint configuration.

        Args:
            joint_pos_deg: Joint positions in degrees (numpy array)
            verbose: If True, print debug info

        Returns:
            4x4 transformation matrix of the end-effector pose
        """

        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_deg[: len(self.joint_names)])
        if verbose:
            print("Input joints (deg):", joint_pos_deg)
            print("Converted joints (rad):", joint_pos_rad)

        # Update joint positions in Placo robot
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, joint_pos_rad[i])
            if verbose:
                print(f"Set joint {joint_name} to {joint_pos_rad[i]:.6f} rad")

        # Update kinematics
        self.robot.update_kinematics()
        if verbose:
            print("Kinematics updated")

        # Optional: print intermediate joint transforms if available
        if hasattr(self.robot, "get_joint_T_world"):
            for joint_name in self.joint_names:
                T = self.robot.get_joint_T_world(joint_name)
                if verbose:
                    pos = T[:3, 3]
                    print(f"Joint {joint_name} world pos: {pos}")

        # Get the transformation matrix of the target frame
        T_ee = self.robot.get_T_world_frame(self.target_frame_name)
        if verbose:
            pos = T_ee[:3, 3]
            print(f"End-effector pose:\n{T_ee}\nPosition: {pos}")

        return T_ee

    def inverse_kinematics(
        self, current_joint_pos, desired_ee_pose, position_weight=1.0, orientation_weight=0.01, verbose=False
    ):
        """
        Compute inverse kinematics using Placo solver with detailed debug outputs.

        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess)
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Weight for position constraint in IK
            orientation_weight: Weight for orientation constraint in IK, set to 0.0 to only constrain position
            verbose: If True, print debug info

        Returns:
            Joint positions in degrees that achieve the desired end-effector pose
        """

        # Convert current joint positions to radians for initial guess
        current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])
        if verbose:
            print("Current joints (deg):", current_joint_pos)
            print("Current joints (rad):", current_joint_rad)

        # Set current joint positions as initial guess
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, current_joint_rad[i])
            if verbose:
                print(f"Initial guess joint {joint_name} set to {current_joint_rad[i]:.6f} rad")

        # Update the target pose for the frame task
        if verbose:
            print("Desired EE pose:\n", desired_ee_pose)
        self.tip_frame.T_world_frame = desired_ee_pose

        # Configure the task
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)
        if verbose:
            print(f"Task configured with position_weight={position_weight}, orientation_weight={orientation_weight}")

        # Solve IK
        
        self.robot.update_kinematics()
        self.solver.solve(True)
        if verbose:
            print("IK solver finished and kinematics updated")

        # Extract joint positions
        joint_pos_rad = []
        for joint_name in self.joint_names:
            joint = self.robot.get_joint(joint_name)
            joint_pos_rad.append(joint)
            if verbose:
                print(f"IK solution joint {joint_name}: {joint:.6f} rad")

        # Convert back to degrees
        joint_pos_deg = np.rad2deg(joint_pos_rad)
        if verbose:
            print("IK solution (deg):", joint_pos_deg)

        # Quick FK check of IK result
        if verbose:
            T_ee_result = self.robot.get_T_world_frame(self.target_frame_name)
            pos_result = T_ee_result[:3, 3]
            print("FK from IK solution position:", pos_result)
            pos_diff = pos_result - desired_ee_pose[:3, 3]
            print("Position difference vs target:", pos_diff)

        if verbose:
            self.solver.dump_status()

        # Preserve gripper position if present
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_deg
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        else:
            return joint_pos_deg


