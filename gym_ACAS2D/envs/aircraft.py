import math

from gym_ACAS2D.settings import *


class Aircraft:

    def __init__(self, x, y, v_air, psi, psi_dot=0, a_lat=0):
        self.x = x
        self.y = y
        self.v_air = v_air
        self.psi = psi
        self.psi_dot = psi_dot
        self.a_lat = a_lat

    def update_state(self):
        # Delta t
        dt = 1 / FPS
        # Rate of heading angle
        self.psi_dot = self.a_lat / (self.v_air * dt)
        # Heading angle
        self.psi = (self.psi + (self.psi_dot * dt)) % 360
        psi_rad = (self.psi / 360.0) * 2 * math.pi
        # Position
        self.x = self.x + (self.v_air * math.cos(psi_rad) * dt)
        self.y = self.y + (self.v_air * math.sin(psi_rad) * dt)

    def out_of_bounds(self, width, height):
        return self.x < 0 or self.x > width or self.y < 0 or self.y > height


class PlayerAircraft(Aircraft):
    pass


class TrafficAircraft(Aircraft):
    pass
