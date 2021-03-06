from gym_ACAS2D.settings import *
from gym_ACAS2D.envs.kinematics import *


def heading_reward(psi, phi, exp=4):
    if (0 <= psi <= 360) and (0 <= phi <= 360):
        return (1 - delta_heading(psi, phi) / 180) ** exp
    else:
        raise ValueError("Heading and relative angle must be in [0, 360].")


def closest_approach_reward(v_closing, d_cpa, exp=4):
    if v_closing > 0:
        return 1
    else:
        return min(1, (d_cpa / SAFE_DISTANCE) ** exp)


def plan_deviation_reward(d_dev, exp=0.5):
    # d_dev can be positive or negative, depending on the side of the deviation (up or down)
    d_dev = abs(d_dev)
    d_goal_init = (WIDTH - GOAL_RADIUS) - (2 * AIRCRAFT_SIZE)
    d_dev_max = d_goal_init / 2
    if d_dev > d_dev_max:
        return 0
    else:
        return (1 - d_dev / d_dev_max) ** exp


def polarized_plan_deviation_reward(d_dev, d_cpa, exp=0.5):
    d_goal_init = (WIDTH - GOAL_RADIUS) - (2 * AIRCRAFT_SIZE)
    d_dev_max = d_goal_init / 2
    # d_dev can be positive or negative, depending on the side of the deviation (up or down)
    # d_cpa can be positive or negative, depending on the rel. position (higher or lower) of player wrt traffic at CPA
    # We want to push the agent's trajectory towards the CPA side
    if (abs(d_dev) > d_dev_max) or (d_cpa * d_dev) < 0:
        return 0
    elif d_cpa >= 0 and d_dev > 0:
        return (1 - d_dev / d_dev_max) ** exp
    else:
        return (1 - abs(d_dev) / d_dev_max) ** exp


def goal_distance_reward(d_goal, exp=4):
    if d_goal >= 0:
        d_goal_init = (WIDTH - GOAL_RADIUS) - (2 * AIRCRAFT_SIZE)
        d_goal_max = d_goal_init + (AIRSPEED / FPS) * MAX_STEPS
        return min(1, (1 - d_goal / d_goal_max) ** exp)
    else:
        raise ValueError("Distance to goal cannot be negative.")


def step_reward_5(v_closing, psi, phi, d_cpa, d_goal, d_dev):
    if v_closing <= 0:
        return heading_reward(psi, phi) * \
               closest_approach_reward(v_closing, d_cpa) * \
               plan_deviation_reward(d_dev)
    else:
        return heading_reward(psi, phi) * \
               goal_distance_reward(d_goal)


def step_reward_6(v_closing, psi, phi, d_cpa, d_goal, d_dev):
    if v_closing <= 0:
        return heading_reward(psi, phi) * \
               closest_approach_reward(v_closing, d_cpa) * \
               polarized_plan_deviation_reward(d_dev, d_cpa)
    else:
        return heading_reward(psi, phi) * \
               goal_distance_reward(d_goal)