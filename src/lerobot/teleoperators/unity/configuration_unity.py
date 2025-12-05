from dataclasses import dataclass

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("unity")
@dataclass
class UnityEndEffectorTeleopConfig(TeleoperatorConfig):
    fps = 30
    teleop_time_s = 180
    mock: bool = False

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    