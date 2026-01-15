from dataclasses import dataclass

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("unity")
@dataclass
class UnityEndEffectorTeleopConfig(TeleoperatorConfig):
    fps: int = 30
    teleop_time_s: int = 180
    mock: bool = False
    unity_projector = None # ??
    display_data: bool = False

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    