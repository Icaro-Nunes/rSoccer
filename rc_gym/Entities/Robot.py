from dataclasses import dataclass

@dataclass
class Robot:
    yellow: bool = None
    id: int = None
    x: float = None
    y: float = None
    z: float = None
    v_x: float = 0
    v_y: float = 0
    v_theta: float = 0
    theta: float = None
    kick_v_x: float = 0
    kick_v_z: float = 0
    dribbler: bool = False
    wheel_speed: bool = False
    v_wheel1: float = 0
    v_wheel2: float = 0
    v_wheel3: float = 0
    v_wheel4: float = 0