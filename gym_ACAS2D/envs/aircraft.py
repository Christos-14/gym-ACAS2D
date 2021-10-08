import random
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
        # Rate of heading angle
        self.psi_dot = self.a_lat / self.v_air
        # Heading angle
        self.psi = self.psi + (self.psi_dot * (1 / FPS))
        psi_rad = (self.psi / 360.0) * 2 * math.pi
        # Position
        self.x = self.x + (self.v_air * math.cos(psi_rad))
        self.y = self.y + (self.v_air * math.sin(psi_rad))

    def out_of_bounds(self, width, height):
        return self.x < 0 or self.x > width or self.y < 0 or self.y > height


class PlayerAircraft(Aircraft):
    pass


class TrafficAircraft(Aircraft):

    def bounce(self, width, height):
        # Make sure the aircraft stays on the screen
        if self.x < 0:
            self.x = 0
        elif self.x + AIRCRAFT_SIZE > width:
            self.x = width - AIRCRAFT_SIZE
        if self.y < 0:
            self.y = 0
        elif self.y + AIRCRAFT_SIZE > height:
            self.y = height - AIRCRAFT_SIZE
        # Update its psi to a random but opposite direction
        h1 = (self.psi + MIN_BOUNCE_ANGLE) % 360
        h2 = (self.psi + MAX_BOUNCE_ANGLE) % 360
        self.psi = random.uniform(min(h1, h2), max(h1, h2))
