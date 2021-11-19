import numpy as np
import math

from gym_ACAS2D.settings import *


def distance(x1, y1, x2, y2):
    # Player and goal positions as np.array
    p1 = np.array((x1, y1))
    p2 = np.array((x2, y2))
    # Euclidean distance between player and goal
    d = np.linalg.norm(p1 - p2, 2)
    return d


def relative_angle(x1, y1, x2, y2):
    # The psi that would lead the player straight to the goal
    dx = x2 - x1
    dy = y2 - y1
    rads = math.atan2(dy, dx) % (2 * math.pi)
    degrees = math.degrees(rads)
    return degrees


def relative_speed(aircraft1, aircraft2):
    # Aircraft 1
    v1 = aircraft1.v_air
    psi1 = aircraft1.psi
    psi1_rad = (psi1 / 360.0) * 2 * math.pi
    # Aircraft 2
    v2 = aircraft2.v_air
    psi2 = aircraft2.psi
    psi2_rad = (psi2 / 360.0) * 2 * math.pi
    # Velocity of AC 1 relative to AC 2
    v12x = v1 * np.cos(psi1_rad) - v2 * np.cos(psi2_rad)
    v12y = v1 * np.sin(psi1_rad) - v2 * np.sin(psi2_rad)
    return v12x, v12y


def distance_closest_approach(aircraft1, aircraft2):
    d = distance(aircraft1.x, aircraft1.y,
                 aircraft2.x, aircraft2.y)
    a_rel = relative_angle(aircraft1.x, aircraft1.y,
                           aircraft2.x, aircraft2.y)
    a_rel_rad = (a_rel / 360.0) * 2 * math.pi
    v12x, v12y = relative_speed(aircraft1, aircraft2)
    h_rel_rad = np.arctan(v12y / v12x)
    dca = d * np.sin(a_rel_rad - h_rel_rad)
    return dca


def closing_speed(aircraft1, aircraft2):
    # Delta t
    dt = 1 / FPS

    # Aircraft 1
    psi_dot_1 = aircraft1.a_lat / aircraft1.v_air
    psi_1 = (aircraft1.psi + (psi_dot_1 * dt)) % 360
    psi_rad_1 = (psi_1 / 360.0) * 2 * math.pi
    x1 = aircraft1.x + (aircraft1.v_air * math.cos(psi_rad_1) * dt)
    y1 = aircraft1.y + (aircraft1.v_air * math.sin(psi_rad_1) * dt)

    p1 = np.array([x1, y1])
    v1 = np.array([aircraft1.v_air * math.cos(psi_rad_1) * dt, aircraft1.v_air * math.sin(psi_rad_1) * dt])

    # Aircraft 2
    psi_dot_2 = aircraft2.a_lat / aircraft2.v_air
    psi_2 = (aircraft2.psi + (psi_dot_2 * dt)) % 360
    psi_rad_2 = (psi_2 / 360.0) * 2 * math.pi
    x2 = aircraft2.x + (aircraft2.v_air * math.cos(psi_rad_2) * dt)
    y2 = aircraft2.y + (aircraft2.v_air * math.sin(psi_rad_2) * dt)

    p2 = np.array([x2, y2])
    v2 = np.array([aircraft2.v_air * math.cos(psi_rad_2) * dt, aircraft1.v_air * math.sin(psi_rad_2) * dt])

    # Closing speed
    c = (np.dot((v1 - v2), (p1 - p2)) / distance(x1, y1, x2, y2)) / dt

    return c


def delta_heading(psi, phi):
    return min(abs(psi - phi), 360 - abs(psi - phi))
