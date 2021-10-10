import numpy as np
import random
import pygame
import math

from gym_ACAS2D.envs.aircraft import PlayerAircraft, TrafficAircraft
from gym_ACAS2D.settings import *


class ACAS2DGame:
    def __init__(self, episode=None, manual=False):

        # Initialize PyGame
        pygame.init()
        # Title and icon
        pygame.display.set_caption(CAPTION)
        pygame.display.set_icon(pygame.image.load(LOGO))

        # Create the screen: WIDTH x HEIGHT
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        # Game clock
        self.clock = pygame.time.Clock()
        # Episode
        self.episode = episode
        # Time steps counter
        self.steps = 0

        # Game status flags
        self.manual = manual  # Is the player controlled manually?
        self.running = True    # Is the game running?
        self.quit = False  # Has the game window been closed?

        # Game outcome
        self.outcome = None

        # Load images
        self.playerIMG = pygame.image.load(PLAYER_IMG)
        self.goalIMG = pygame.image.load(GOAL_IMG)
        self.trafficIMG = pygame.image.load(TRAFFIC_IMG)
        # Text font
        self.font = pygame.font.Font(FONT_NAME, FONT_SIZE)

        # Set the goal position at random, in the top part of the airspace.
        self.goal_x = random.uniform(AIRCRAFT_SIZE, WIDTH - AIRCRAFT_SIZE)
        self.goal_y = random.uniform(AIRCRAFT_SIZE, HEIGHT / 5)

        # Set the player starting position at random, in the bottom part of the airspace
        # Set the player starting psi and v_air
        player_x = random.uniform(0, WIDTH - AIRCRAFT_SIZE)
        player_y = random.uniform(4 * HEIGHT / 5, HEIGHT - AIRCRAFT_SIZE)
        player_speed = AIRSPEED
        self.player = PlayerAircraft(x=player_x, y=player_y, v_air=player_speed, psi=0)
        # Set the initial heading towards the goal; Assumption is that a path has been provided to the agent.
        self.player.psi = self.heading_to_goal()

        # Number of traffic aircraft
        self.num_traffic = random.randint(MIN_TRAFFIC, MAX_TRAFFIC)

        # Set the traffic aircraft positions, headings and speeds at random, in the middle part of the airspace.
        self.traffic = []
        for t in range(self.num_traffic):
            # Random position in the mid part of the airspace
            t_x = random.uniform(0, WIDTH - AIRCRAFT_SIZE)
            t_y = random.uniform(0, 3 * HEIGHT / 5)
            # Random v_air
            t_speed = random.uniform(AIRSPEED_FACTOR_MIN, AIRSPEED_FACTOR_MAX) * AIRSPEED
            # Random psi: 0..360 degrees
            t_heading = random.uniform(0, 360)
            self.traffic.append(TrafficAircraft(x=t_x, y=t_y, v_air=t_speed, psi=t_heading))

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
        pl = np.array((self.player.x, self.player.y))
        gl = np.array((self.goal_x, self.goal_y))
        # Euclidean distance between player and goal
        d = np.linalg.norm(pl - gl, 2)
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
            if d < COLLISION_RADIUS:
                collision = True
                break
        return collision

    def check_out_of_bounds(self):
        # Player and goal positions as np.array
        out_of_bounds = (self.player.x < 0) \
                        or (self.player.x > WIDTH) \
                        or (self.player.y < 0) \
                        or (self.player.y > HEIGHT)
        return out_of_bounds

    def check_goal(self):
        # Player and goal positions as np.array
        pl = np.array((self.player.x, self.player.y))
        gl = np.array((self.goal_x, self.goal_y))
        # Euclidean distance between player and goal
        d = np.linalg.norm(pl-gl, 2)
        # If the distance is less than the collision radius, player reached the goal
        goal = d < GOAL_RADIUS
        return goal

    def heading_to_goal(self):
        # The psi that would lead the player straight to the goal
        dx = self.goal_x - self.player.x
        dy = self.goal_y - self.player.y
        rads = math.atan2(dy, dx) % (2*math.pi)
        degs = math.degrees(rads)
        return degs

    def observe(self):

        # Increase number of steps in the game (all steps start with an observation)
        self.steps += 1

        # Observation: Dictionary of  4 vectors of shape (MAX_TRAFFIC+2, )
        # Keys -> x, y, v_air, psi
        # Values ordering -> [player|goal|traffic|padding]
        obs = {}

        # Player and goal
        obs_x = [self.player.x, self.goal_x]
        obs_y = [self.player.y, self.goal_y]
        obs_v_air = [self.player.v_air, 0]
        obs_psi = [self.player.psi, 0]
        # Traffic aircraft
        for t in self.traffic:
            obs_x.append(t.x)
            obs_y.append(t.y)
            obs_v_air.append(t.v_air)
            obs_psi.append(t.psi)
        # Padding with zeros
        obs_x += ([0]*(MAX_TRAFFIC-self.num_traffic))
        obs_y += ([0]*(MAX_TRAFFIC-self.num_traffic))
        obs_v_air += ([0]*(MAX_TRAFFIC-self.num_traffic))
        obs_psi += ([0]*(MAX_TRAFFIC-self.num_traffic))

        # Construct observation dict
        obs["x"] = np.array(obs_x).astype(np.float64)
        obs["y"] = np.array(obs_y).astype(np.float64)
        obs["v_air"] = np.array(obs_v_air).astype(np.float64)
        obs["psi"] = np.array(obs_psi).astype(np.float64)

        return obs

    def action(self, action):
        # Update player a_lat based on action taken
        # Action is scaled to [-1, 1] ; scale to original [-ACC_LAT_LIMIT, ACC_LAT_LIMIT]
        self.player.a_lat = action[0] * ACC_LAT_LIMIT
        # Update player position based on that v_air and psi
        self.player.update_state()
        # If the game is still running, update the traffic aircraft positions.
        for t in self.traffic:
            if self.running:
                t.update_state()
                if t.out_of_bounds(WIDTH, HEIGHT):
                    t.bounce(WIDTH, HEIGHT)

    def evaluate(self):
        reward = 0
        # Penalise time spent
        reward += REWARD_STEP
        # # Reward min_separation maintained
        # reward += REWARD_MIN_SEPARATION_FACTOR * self.minimum_separation()
        # Penalise collisions.
        if self.detect_collisions():
            reward += REWARD_COLLISION
        # Reward reaching the goal
        if self.check_goal():
            reward += REWARD_GOAL
        return reward

    def is_done(self):
        # Check for Timeout: if max number of steps has been reached
        if self.steps == MAX_STEPS:
            self.running = False
            self.outcome = 3
            return True
        # Check for collisions
        if self.detect_collisions():
            self.running = False
            self.outcome = 2
            return True
        # Check if we have reached the goal
        if self.check_goal():
            self.running = False
            self.outcome = 1
            return True
        # Otherwise, the game is on
        return False

    def view(self):
        # Detect events
        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                self.quit = True

        # Change background colour to sky colour RGB value
        self.screen.fill(SKY_RGB)

        # Place player in the game
        self.screen.blit(self.playerIMG, (self.player.x - (AIRCRAFT_SIZE / 2),
                                          self.player.y - (AIRCRAFT_SIZE / 2)))

        # Place goal in the game
        self.screen.blit(self.goalIMG, (self.goal_x - (AIRCRAFT_SIZE / 2),
                                        self.goal_y - (AIRCRAFT_SIZE / 2)))

        # Place traffic aircraft in the game
        for t in self.traffic:
            self.screen.blit(self.trafficIMG, (t.x - (AIRCRAFT_SIZE / 2),
                                               t.y - (AIRCRAFT_SIZE / 2)))

        # Draw collision circle around aircraft
        pygame.draw.circle(self.screen, GREEN_RGB, (self.player.x, self.player.y),
                           COLLISION_RADIUS, 1)
        # for t in self.traffic:
        #     pygame.draw.circle(self.screen, RED_RGB, (t.x, t.y), COLLISION_RADIUS, 1)

        # Display minimum separation
        min_separation = self.minimum_separation()
        ms = self.font.render("Min. Separation: {}".format(round(min_separation, 1)), True, FONT_RGB)
        self.screen.blit(ms, (20, HEIGHT - 20))

        # Display episode and 'time' (number of game loop iterations)
        st = self.font.render("Steps: {}".format(self.steps), True, FONT_RGB)
        self.screen.blit(st, (WIDTH / 2 - 50, HEIGHT - 20))
        ep = self.font.render("Episode: {}".format(self.episode), True, FONT_RGB)
        self.screen.blit(ep, (WIDTH / 2 - 50, HEIGHT - 40))

        # Display distance to target
        dist_to_goal = self.distance_to_goal()
        dg = self.font.render("Distance to goal: {}".format(round(dist_to_goal, 1)), True, FONT_RGB)
        self.screen.blit(dg, (WIDTH - 200, HEIGHT - 20))

        # Detect collisions
        if self.detect_collisions():
            mes = self.font.render("Collision!", True, FONT_RGB)
            self.screen.blit(mes, (WIDTH / 2 - 30, HEIGHT / 2))

        # Check if player reached the goal
        if self.check_goal():
            mes = self.font.render("Goal reached!", True, FONT_RGB)
            self.screen.blit(mes, (WIDTH / 2 - 40, HEIGHT / 2))

        # Update the game screen
        pygame.display.update()
