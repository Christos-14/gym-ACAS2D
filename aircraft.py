import random
import math


class Aircraft:

    def __init__(self, size, x=None, y=None, speed=None, heading=None):
        self.size = size
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
        elif self.x + self.size > width:
            self.x = width - self.size
        if self.y < 0:
            self.y = 0
        elif self.y + self.size > height:
            self.y = height - self.size
        # Update its heading to a random but opposite direction
        h1 = (self.heading + 120) % 360
        h2 = (self.heading + 240) % 360
        self.heading = random.randint(min(h1, h2), max(h1, h2))
