import numpy as np
import random
import pygame
import math

from gym_ACAS2D.envs.aircraft import PlayerAircraft, TrafficAircraft
from gym_ACAS2D.settings import *


class ACAS2DGame:
    def __init__(self, episode=None, manual=False, verbose=False):

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
        # Total reward
        self.total_reward = 0

        # Game status flags
        self.manual = manual  # Is the player controlled manually?
        self.running = True  # Is the game running?
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

        # Maximum allowed distance from goal
        self.max_distance = np.sqrt(WIDTH**2 + HEIGHT**2)

        # Set the player starting position at random, in the bottom part of the airspace
        # Set the player starting psi and v_air
        player_x = random.uniform(0, WIDTH - AIRCRAFT_SIZE)
        player_y = random.uniform(4 * HEIGHT / 5, HEIGHT - AIRCRAFT_SIZE)
        player_speed = AIRSPEED
        player_psi = random.uniform(0, 360)
        self.player = PlayerAircraft(x=player_x, y=player_y, v_air=player_speed, psi=player_psi)

        # Set player's initial heading towards the general direction of the goal.
        self.player.psi = self.heading_to_goal() + random.uniform(-INITIAL_HEADING_LIM, INITIAL_HEADING_LIM)

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
        if self.num_traffic == 0:
            return float("inf")
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

    def check_timeout(self):
        return self.steps == MAX_STEPS

    def check_too_far(self):
        return self.distance_to_goal() > self.max_distance

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

    def check_goal(self):
        # Player and goal positions as np.array
        pl = np.array((self.player.x, self.player.y))
        gl = np.array((self.goal_x, self.goal_y))
        # Euclidean distance between player and goal
        d = np.linalg.norm(pl - gl, 2)
        # If the distance is less than the collision radius, player reached the goal
        goal = d < GOAL_RADIUS
        return goal

    def heading_to_goal(self):
        # The psi that would lead the player straight to the goal
        dx = self.goal_x - self.player.x
        dy = self.goal_y - self.player.y
        rads = math.atan2(dy, dx) % (2 * math.pi)
        degrees = math.degrees(rads)
        return degrees

    def observe(self):

        # Increase number of steps in the game (all steps start with an observation)
        self.steps += 1

        # Observation: Dictionary of  4 vectors of shape (MAX_TRAFFIC+2, )
        # Keys -> x, y, v_air, psi
        # Values ordering -> [player|goal|traffic|padding]
        obs = {}

        # Current time step, normalised (from 0..MAX_STEPS to 0..1)
        obs_time = [self.steps/MAX_STEPS]

        # Goal position, normalised
        obs_goal = [self.goal_x/WIDTH, self.goal_y/HEIGHT]

        # Player position, airspeed and heading, normalised
        obs_player = [self.player.x/WIDTH, self.player.y/HEIGHT,
                      self.player.v_air/(AIRSPEED_FACTOR_MAX * AIRSPEED),
                      self.player.psi/360]

        # Traffic positions, airspeeds and headings, normalised
        obs_traffic = []
        for t in self.traffic:
            obs_traffic.append(t.x/WIDTH)
            obs_traffic.append(t.y/HEIGHT)
            obs_traffic.append(t.v_air/(AIRSPEED_FACTOR_MAX * AIRSPEED))
            obs_traffic.append(t.psi/360)
        # Padding with zeros
        obs_traffic += [0] * (4 * (MAX_TRAFFIC - self.num_traffic))

        # Construct observation dict
        obs["time"] = np.array(obs_time).astype(np.float64)
        obs["goal"] = np.array(obs_goal).astype(np.float64)
        obs["player"] = np.array(obs_player).astype(np.float64)
        obs["traffic"] = np.array(obs_traffic).astype(np.float64)

        # print("observe() 	>>> obs: {}".format(obs))
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
        # print("action() 	>>> Action: {}".format(action))

    def evaluate(self):
        reward = 0
        # Time discount factor
        tdf = 1 - (self.steps / MAX_STEPS)
        # Time discounted distance reward
        d = self.distance_to_goal()
        d_max = self.max_distance
        r_dist = 1 - (d / d_max) ** 0.4
        reward += r_dist * tdf
        if self.check_too_far():
            reward += REWARD_TOO_FAR
        # Penalise timeouts.
        if self.check_timeout():
            reward += REWARD_TIMEOUT
        # Penalise collisions.
        if self.detect_collisions():
            reward += REWARD_COLLISION
        # Reward reaching the goal
        if self.check_goal():
            reward += REWARD_GOAL
        # Accumulate episode rewards
        self.total_reward += reward
        # print("evaluate() 	>>> Reward: {}".format(reward))
        return reward

    def is_done(self):
        done = False
        # Check for Too Far
        if self.check_too_far():
            self.running = False
            self.outcome = 4
            done = True
        # Check for Timeout
        elif self.check_timeout():
            self.running = False
            self.outcome = 3
            done = True
        # Check for collisions
        elif self.detect_collisions():
            self.running = False
            self.outcome = 2
            done = True
        # Check if we have reached the goal
        elif self.check_goal():
            self.running = False
            self.outcome = 1
            done = True
        if done:
            print("is_done() 	>>> Outcome: {}".format(self.outcome))
        return done

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

        # Draw goal radius around goal
        pygame.draw.circle(self.screen, RED_RGB, (self.goal_x, self.goal_y),
                           GOAL_RADIUS, 1)

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
