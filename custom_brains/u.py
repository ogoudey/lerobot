from lerobot.teleoperators.unity.configuration_unity import (
    UnityEndEffectorTeleopConfig,
)


if __name__ == "__main__":
    cfg = UnityEndEffectorTeleopConfig()
    ut = UnityEndEffectorTeleop(cfg)
    ut.connect()
