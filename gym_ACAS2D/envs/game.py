import numpy as np
import random
import pygame
import math

from gym_ACAS2D.envs.aircraft import PlayerAircraft, TrafficAircraft
from gym_ACAS2D.settings import *


class ACAS2DGame:
    def __init__(self, manual=False):

        # Initialize PyGame
        pygame.init()
        # Title and icon
        pygame.display.set_caption(CAPTION)
        pygame.display.set_icon(pygame.image.load(LOGO))

        # Create the screen: WIDTH x HEIGHT
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        # Game clock
        self.clock = pygame.time.Clock()
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
        self.goal_x = random.randint(AIRCRAFT_SIZE, WIDTH - AIRCRAFT_SIZE)
        self.goal_y = random.randint(AIRCRAFT_SIZE, round(HEIGHT / 5))

        # Set the player starting position at random, in the bottom part of the airspace
        # Set the player starting psi and v_air
        player_x = random.randint(0, WIDTH - AIRCRAFT_SIZE)
        player_y = random.randint(round(4 * HEIGHT / 5), HEIGHT - AIRCRAFT_SIZE)
        player_speed = AIRSPEED
        self.player = PlayerAircraft(x=player_x, y=player_y, v_air=player_speed, psi=0)
        # Set the initial heading towards the goal; Assumption is that a path has been provided to the agent.
        self.player.psi = self.heading_to_goal()

        # Set the traffic aircraft positions, headings and speeds at random, in the middle part of the airspace.
        self.traffic = []
        for t in range(N_TRAFFIC):
            # Random position in the mid part of the airspace
            t_x = random.randint(0, WIDTH - AIRCRAFT_SIZE)
            t_y = random.randint(0, round(3 * HEIGHT / 5))
            # Random v_air: low (75%), medium (100%), or high (125%)
            t_speed = AIRSPEED
            # Random psi: 0..359 degrees
            t_heading = random.randint(0, 360)
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
                self.running = False
                self.win = False
                break
        return collision

    def check_out_of_bounds(self):
        # Player and goal positions as np.array
        out_of_bounds = (self.player.x < 0) \
                        or (self.player.x > WIDTH) \
                        or (self.player.y < 0) \
                        or (self.player.y > HEIGHT)
        return out_of_bounds

    def check_goal(self, d_reached=20):
        # Player and goal positions as np.array
        pl = np.array((self.player.x, self.player.y))
        gl = np.array((self.goal_x, self.goal_y))
        # Euclidean distance between player and goal
        d = np.linalg.norm(pl-gl, 2)
        # If the distance is less than the collision radius, player reached the goal
        goal = d < d_reached
        if goal:
            self.running = False
            self.win = True
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

        # Player position
        pos = [self.player.x, self.player.y]
        # Traffic positions
        for t in self.traffic:
            pos.append(t.x)
            pos.append(t.y)
        # Goal position
        pos.append(self.goal_x)
        pos.append(self.goal_y)

        # Observation
        obs = np.array(pos).astype(np.float32)

        return obs

    def action(self, action):
        # Update player a_lat based on action taken
        self.player.a_lat = action[0]
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
        # Penalise time spent to reach goal
        reward += REWARD_STEP
        # Reward min_separation maintained
        reward += REWARD_MIN_SEPARATION_FACTOR * self.minimum_separation()
        # Reward reaching the goal
        if self.check_goal():
            reward += REWARD_GOAL
        # Penalise collisions and getting out of bounds.
        if self.detect_collisions() or self.check_out_of_bounds():
            reward += REWARD_COLLISION
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
        ms = self.font.render("Min. Separation: {}".format(round(min_separation, 3)), True, FONT_RGB)
        self.screen.blit(ms, (20, HEIGHT - 20))

        # Display 'time' (number of game loop iterations)
        st = self.font.render("Steps: {}".format(self.steps), True, FONT_RGB)
        self.screen.blit(st, (round(WIDTH / 2) - 50, HEIGHT - 20))

        # Display distance to target
        dist_to_goal = self.distance_to_goal()
        dg = self.font.render("Distance to goal: {}".format(round(dist_to_goal, 3)), True, FONT_RGB)
        self.screen.blit(dg, (WIDTH - 200, HEIGHT - 20))

        # Detect collisions
        if self.detect_collisions():
            mes = self.font.render("Collision!", True, FONT_RGB)
            self.screen.blit(mes, (round(WIDTH / 2) - 30, round(HEIGHT / 2)))

        # Check if player reached the goal
        if self.check_goal():
            mes = self.font.render("Goal reached!", True, FONT_RGB)
            self.screen.blit(mes, (round(WIDTH / 2) - 40, round(HEIGHT / 2)))

        # Update the game screen
        pygame.display.update()
