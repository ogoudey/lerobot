from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("kinova_gen3_end_effector")
@dataclass
class KinovaGen3EndEffectorConfig(RobotConfig):
    # Port to connect to the arm
    
    urdf_path: str | None = None

    # End-effector frame name in the URDF
    target_frame_name: str = "gripper_frame_link_name_for_kinova"

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)