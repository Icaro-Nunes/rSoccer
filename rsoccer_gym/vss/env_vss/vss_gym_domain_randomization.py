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

        self.params_intervals = {
            "robot_body_mass": (0.100, 0.300),
            "robot_wheel_motor_max_rpm": (390.0, 520.0),
            "robot_wheel_perpendicular_friction": (0.7, 1.0),
            "robot_wheel_tangent_friction": (0.6, 0.95)
        }

        super().__init__(self.generate_params())

    def generate_params(self):
        params = self.default_params.copy()
        for key in params:
            low, high = self.params_intervals[key]
            params[key] = np.random.uniform(low, high)
        return params

    def reset(self):
        return super().reset(self.generate_params())