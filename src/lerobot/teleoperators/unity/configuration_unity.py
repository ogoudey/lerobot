from dataclasses import dataclass

from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("unity")
@dataclass
class UnityEndEffectorTeleopConfig(TeleoperatorConfig):
    mock: bool = False