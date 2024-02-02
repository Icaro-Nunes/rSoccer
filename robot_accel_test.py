from math import atan2, sqrt
from typing import Tuple
import gym
import rsoccer_gym
import numpy as np

env = gym.make("VSS-v0")

rbt_motor_max_rpm = 440.0
max_wheel_rad_s = (rbt_motor_max_rpm/60) * 2 * np.pi
rbt_wheel_radius = 0.026
max_v = max_wheel_rad_s * rbt_wheel_radius

rbt_radius = 0.0375
l = rbt_radius

max_w = max_v/l

field_length = 1.5
field_width = 1.3
penalty_length = 0.15
time_step = 0.025

max_pos = max(field_width / 2, (field_length / 2) 
                                + penalty_length)

NORM_BOUNDS = 1.2

# desired velocity (in cm/s) to vwheel pair (m/s, m/s)
def v_to_wheel(v: int) -> Tuple[float, float]:
  v_m_s = float(v)/100
  v_m_s = np.clip(v_m_s, -max_v, max_v)

  vr = v_m_s
  vl = v_m_s
  
  wr = vr/rbt_wheel_radius
  wl = vl/rbt_wheel_radius

  return wl, wr 

def w_to_wheel(w: float) -> Tuple[float, float]:
  w = np.clip(w, -max_w, max_w)

  vr = w*l
  vl = -w*l
  
  wr = vr/rbt_wheel_radius
  wl = vl/rbt_wheel_radius

  return wl, wr 

def vwheel_to_action(vw: float) -> float:
    return float(vw)/max_v

def extract_robot_from_state(state: np.ndarray) -> Tuple[float, float, float, float, float]:
    x_n = state[4]
    y_n = state[5]
    sin_th = state[6]
    cos_th = state[7]
    vx_n = state[8]
    vy_n = state[9]
    w_n = state[10]

    x = x_n*max_pos #in meters
    y = y_n*max_pos #in meters
    vx = vx_n*max_v #in meters/second
    vy = vy_n*max_v #in meters/second
    w = w_n*max_w #in rads/second
    theta = atan2(sin_th, cos_th)

    v = sqrt(vx**2 + vy**2)

    # as   mm      mm      rads   mm/s    rads/s
    return x*1000, y*1000, theta, v*1000, w


log_file = open("sim_accels.csv", "w")
log_file.write("TIME, COMMAND_V, COMMAND_W, ROBOT_X, ROBOT_Y, ROBOT_THETA, ROBOT_V, ROBOT_W\n")

steps = 0

def log(state: np.ndarray, command_v: int = 0, command_w: float = 0):
    x, y, theta, v, w = extract_robot_from_state(state)

    log_file.write(
      f"{steps*time_step}, {command_v}, {command_w}, {x}, {y}, {theta}, {v}, {w}\n"
    )
    log_file.flush()

# back and forth
for v in range(10, 110, 10):
    state = env.reset()

    log(state)

    steps_1sec = int(1.0/time_step)
    steps_200ms = int(0.2/time_step)

    action_f = [vwheel_to_action(i) for i in  v_to_wheel(v)]
    action_b = [(-i) for i in action_f]
    halt = [0.0, 0.0]

    # forth
    for _ in range(steps_1sec):
        state, r, d, i = env.step(action_f)
        steps+=1
        log(state, command_v=v)
    # pre_back
    for _ in range(steps_200ms):
        state, r, d, i = env.step(halt)
        steps+=1
        log(state, command_v=0)
    # back
    for _ in range(steps_1sec):
        state, r, d, i = env.step(action_b)
        steps+=1
        log(state, command_v=-v)
    # stop
    for _ in range(steps_200ms):
        state, r, d, i = env.step(halt)
        steps+=1
        log(state, command_v=0)



