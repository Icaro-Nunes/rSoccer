import numpy as np
from rsoccer_gym.vss.env_vss.vss_gym import VSSEnv

class VSSDomainRandomizationEnv(VSSEnv):
    def __init__(self):
        self.default_params = {
            "robot_body_mass": 0.120,
            "robot_wheel_motor_max_rpm": 440.0,
            "robot_wheel_perpendicular_friction": 1.0,
            "robot_wheel_tangent_friction": 0.8
        }

        self.params = self.default_params.copy()

        super().__init__(self.generate_params())

    def generate_params(self):
        for key in self.params:
            val = self.default_params[key]
            scale = val/10
            self.params[key] = np.random.normal(val, scale)
        return self.params

    def reset(self):
        return super().reset(self.generate_params())