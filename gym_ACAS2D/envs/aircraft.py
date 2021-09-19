import random
import math

import gym_ACAS2D.settings as settings


class Aircraft:

    def __init__(self, x=None, y=None, speed=None, heading=None):
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = heading

    def update_position(self):
        heading_rad = (self.heading / 360.0) * 2 * math.pi
        self.x = self.x + (self.speed * math.cos(heading_rad))
        self.y = self.y + (self.speed * math.sin(heading_rad))

    def out_of_bounds(self, width, height):
        return self.x < 0 or self.x > width or self.y < 0 or self.y > height


class PlayerAircraft(Aircraft):
    pass


class TrafficAircraft(Aircraft):

    def bounce(self, width, height):
        # Make sure the aircraft stays on the screen
        if self.x < 0:
            self.x = 0
        elif self.x + settings.AIRCRAFT_SIZE > width:
            self.x = width - settings.AIRCRAFT_SIZE
        if self.y < 0:
            self.y = 0
        elif self.y + settings.AIRCRAFT_SIZE > height:
            self.y = height - settings.AIRCRAFT_SIZE
        # Update its heading to a random but opposite direction
        h1 = (self.heading + settings.MIN_BOUNCE_ANGLE) % 360
        h2 = (self.heading + settings.MAX_BOUNCE_ANGLE) % 360
        self.heading = random.randint(min(h1, h2), max(h1, h2))
