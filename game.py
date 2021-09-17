from aircraft import PlayerAircraft, TrafficAircraft
import numpy as np
import random


class Game:
    def __init__(self, width, height, n_traffic, aircraft_size, collision_radius, medium_speed,
                 static=False,
                 manual=False):

        # Airspace dimensions
        self.width = width
        self.height = height
        # Aircraft size
        self.aircraft_size = aircraft_size
        # Collision radius
        self.collision_radius = collision_radius
        # Speed unit
        self.medium_speed = medium_speed
        # Flags
        self.running = True    # Is the game running?
        self.win = None    # Did the player win or lose?
        self.manual = manual  # Is the player controlled manually?

        # Game mode: True for static, False for random
        self.static = static

        if not static:
            # Set the player starting position at random, in the bottom part of the airspace
            # Set the player starting heading and speed to zero
            player_x = random.randint(0, self.width - self.aircraft_size)
            player_y = random.randint(int(3 * self.height / 4), self.height - self.aircraft_size)
            self.player = PlayerAircraft(aircraft_size, x=player_x, y=player_y, speed=0, heading=0)

            # Set the traffic aircraft positions, headings and speeds at random, in the middle part of the airspace.
            self.traffic = []
            for t in range(n_traffic):
                # Random position in the mid part of the airspace
                t_x = random.randint(0, self.width - self.aircraft_size)
                t_y = random.randint(int(self.height / 4), int(2 * self.height / 3))
                # Random speed: low (75%), medium (100%), or high (125%)
                t_speed = (random.randint(3, 6) / 4) * self.medium_speed
                # Random heading: 0..359 degrees
                t_heading = random.randint(0, 360)
                self.traffic.append(TrafficAircraft(aircraft_size, x=t_x, y=t_y, speed=t_speed, heading=t_heading))

            # Set the goal position at random, in the top part of the airspace.
            self.goal_x = random.randint(0, self.width - self.aircraft_size)
            self.goal_y = random.randint(0, int(self.height / 4))

        else:
            raise NotImplementedError

    def minimum_separation(self):
        distances = []
        for t_air in self.traffic:
            # Player and traffic aircraft positions as np.array
            p = np.array((self.player.x, self.player.y))
            t = np.array((t_air.x, t_air.y))
            # Euclidean distance between player and goal
            d = np.linalg.norm(p - t, 2)
            distances.append(d)
        return np.min(distances)

    def distance_to_goal(self):
        # Player and goal positions as np.array
        p = np.array((self.player.x, self.player.y))
        g = np.array((self.goal_x, self.goal_y))
        # Euclidean distance between player and goal
        d = np.linalg.norm(p - g, 2)
        return d

    def detect_collisions(self):
        collision = False
        for t_air in self.traffic:
            # Player and traffic aircraft positions as np.array
            p = np.array((self.player.x, self.player.y))
            t = np.array((t_air.x, t_air.y))
            # Euclidean distance between player and goal
            d = np.linalg.norm(p - t, 2)
            # If the distance is less than the collision radius, player reached the goal
            if d < self.collision_radius:
                collision = True
                self.running = False
                self.win = False
                break
        return collision

    def check_goal(self):
        # Player and goal positions as np.array
        p = np.array((self.player.x, self.player.y))
        g = np.array((self.goal_x, self.goal_y))
        # Euclidean distance between player and goal
        d = np.linalg.norm(p-g, 2)
        # If the distance is less than the collision radius, player reached the goal
        if d < self.collision_radius:
            self.running = False
            self.win = True
        return d < self.collision_radius
