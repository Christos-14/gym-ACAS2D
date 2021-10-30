import numpy as np
import random
import pygame
import math

from gym_ACAS2D.envs.aircraft import PlayerAircraft, TrafficAircraft
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
    h_rel_rad = np.arctan(v12y/v12x)
    dca = d * np.sin(a_rel_rad-h_rel_rad)
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
    c = np.dot((v1 - v2), (p1 - p2)) / distance(x1, y1, x2, y2)

    return c


# def distance_reward(d, exp=0.5):
#     if d >= 0:
#         d_goal_init = (WIDTH - GOAL_RADIUS) - (2 * AIRCRAFT_SIZE)
#         return max(0, 1 - (d / d_goal_init) ** exp)
#     else:
#         raise ValueError("Distance to goal cannot be negative.")


# def closest_approach_reward(d_sep, d_cpa):
#     if (0 <= d_sep <= SAFE_RADIUS) and (0 <= d_cpa <= SAFE_RADIUS):
#         return (d_cpa / SAFE_RADIUS) ** 2
#     else:
#         return 1


def delta_heading(psi, phi):
    return min(abs(psi-phi), abs(psi-phi-360))


def heading_reward(psi, phi):
    if (0 <= psi <= 360) and (0 <= phi <= 360):
        return (1 - delta_heading(psi, phi)/180) ** 4
    else:
        raise ValueError("Heading and relative angle must be in [0, 360].")


def closing_speed_reward(c, exp=2):
    if c > 0:
        return 1
    else:
        c_max = 2 * (AIRSPEED / FPS)
        return max(0, 1 - (abs(c)/c_max) ** exp)


# def separation_reward(s, c, exp=2):
#     if s >= 0:
#         return min(1, (s / (2 * SAFE_RADIUS)) ** exp)
#     else:
#         raise ValueError("Separation cannot be negative.")


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
        self.goal_x = WIDTH - GOAL_RADIUS
        self.goal_y = HEIGHT / 2

        # # Maximum allowed distance from goal
        # self.max_distance = np.sqrt(WIDTH ** 2 + HEIGHT ** 2)

        # Set the player starting position at random, in the bottom part of the airspace
        # Set the player starting psi and v_air
        player_x = COLLISION_RADIUS
        player_y = HEIGHT / 2
        player_speed = AIRSPEED
        player_psi = random.uniform(0, 360)
        self.player = PlayerAircraft(x=player_x, y=player_y, v_air=player_speed, psi=player_psi)
        # Set player's initial heading towards the general direction of the goal.
        self.player.psi = (relative_angle(self.player.x, self.player.y, self.goal_x, self.goal_y) +
                           random.uniform(-INITIAL_HEADING_LIM, INITIAL_HEADING_LIM)) % 360

        # Initial distance to goal
        self.d_goal_initial = self.distance_to_goal()
        # Max possible distance from goal
        self.d_goal_max = self.distance_to_goal() + (AIRSPEED / FPS) * MAX_STEPS
        # Maximum possible separation
        self.d_separation_max = np.sqrt(WIDTH**2 + HEIGHT**2) + (2 * (AIRSPEED / FPS) * MAX_STEPS)

        # Number of traffic aircraft
        self.num_traffic = random.randint(MIN_TRAFFIC, MAX_TRAFFIC)

        # Set the traffic aircraft positions, headings and speeds at random, in the middle part of the airspace.
        self.traffic = []
        for t in range(self.num_traffic):

            if t == 0:
                starts_down = random.randint(0, 1)
                # Random position in the mid part of the airspace
                t_x = WIDTH - COLLISION_RADIUS
                t_y = COLLISION_RADIUS + (starts_down * (HEIGHT - (2 * COLLISION_RADIUS)))
                # Random v_air
                t_speed = random.uniform(AIRSPEED_FACTOR_MIN, AIRSPEED_FACTOR_MAX) * AIRSPEED
                # Random psi: 0..360 degrees
                t_heading = random.uniform(110, 160) + (starts_down * 90)
                self.traffic.append(TrafficAircraft(x=t_x, y=t_y, v_air=t_speed, psi=t_heading))
            else:
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
        distances = [distance(self.player.x, self.player.y, t.x, t.y) for t in self.traffic]
        return np.min(distances)

    def distance_to_goal(self):
        return distance(self.player.x, self.player.y, self.goal_x, self.goal_y)

    def heading_to_goal(self):
        # The psi that would lead the player straight to the goal
        return relative_angle(self.player.x, self.player.y, self.goal_x, self.goal_y)

    def check_timeout(self):
        return self.steps == MAX_STEPS

    def detect_collisions(self):
        for t in self.traffic:
            if distance(self.player.x, self.player.y, t.x, t.y) < 2 * COLLISION_RADIUS:
                return True
        return False

    def check_goal(self):
        return self.distance_to_goal() < GOAL_RADIUS

    def observe(self):

        # Increase number of steps in the game (all steps start with an observation)
        self.steps += 1

        obs = [self.steps / MAX_STEPS,
               self.player.x / WIDTH,
               self.player.y / HEIGHT,
               self.player.psi / 360,
               self.distance_to_goal() / self.d_goal_max,
               self.heading_to_goal() / 360]

        for t in self.traffic:
            x1, y1 = self.player.x, self.player.y
            x2, y2 = t.x, t.y
            obs.append(distance(x1, y1, x2, y2) / self.d_separation_max)
            obs.append(relative_angle(x1, y1, x2, y2) / 360)

        # Padding
        obs += [0] * (2 * (MAX_TRAFFIC - self.num_traffic))

        obs = np.array(obs).astype(np.float64)

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

        # print("action() 	>>> Action: {}".format(action))

    def evaluate(self):

        # reward = 0

        # d_separation = self.minimum_separation()
        v_closing = closing_speed(self.player, self.traffic[0])
        psi = self.player.psi
        phi = relative_angle(self.player.x, self.player.y, self.goal_x, self.goal_y)

        r_step = heading_reward(psi, phi) * closing_speed_reward(v_closing)

        # Time discount factor
        tdf = 1 - (self.steps / MAX_STEPS)
        reward = r_step * tdf

        # # Time discounted distance reward
        # d_goal = self.distance_to_goal()
        # d_init = self.d_goal_initial
        # r_dist = np.tanh((d_init - d_goal) / self.d_goal_max)
        # # x_max = (AIRSPEED / FPS) * MAX_STEPS
        # # r_dist = np.tanh(self.player.x / x_max)
        # reward += r_dist * tdf
        # # Penalise running away
        # if self.check_runaway():
        #     reward += REWARD_RUNAWAY_FACTOR * d_goal

        # # Penalise timeouts.
        # if self.check_timeout():
        #     reward += REWARD_TIMEOUT

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
        # Check for Timeout
        if self.check_timeout():
            self.running = False
            done = True
            self.outcome = 3
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
            print("is_done() 	>>> Outcome: {} Total Reward: {}".format(OUTCOME_NAMES[self.outcome], self.total_reward))
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

        # Draw collision radius around aircraft
        pygame.draw.circle(self.screen, RED_RGB, (self.player.x, self.player.y), COLLISION_RADIUS, 1)
        # pygame.draw.circle(self.screen, YELLOW_RBG, (self.player.x, self.player.y), DANGER_RADIUS, 1)
        # pygame.draw.circle(self.screen, GREEN_RGB, (self.player.x, self.player.y), SAFE_RADIUS, 1)

        # Draw goal radius around goal
        pygame.draw.circle(self.screen, YELLOW_RBG, (self.goal_x, self.goal_y),
                           GOAL_RADIUS, 1)

        # Draw collision radius around traffic aircraft
        for t in self.traffic:
            pygame.draw.circle(self.screen, RED_RGB, (t.x, t.y), COLLISION_RADIUS, 1)
            # pygame.draw.circle(self.screen, YELLOW_RBG, (t.x, t.y), DANGER_RADIUS, 1)
            # pygame.draw.circle(self.screen, GREEN_RGB, (t.x, t.y), SAFE_RADIUS, 1)

        # Display distance/separation
        d_goal = self.distance_to_goal()
        dg = self.font.render("Distance to goal: {}".format(round(d_goal, 1)), True, BLACK_RGB)
        self.screen.blit(dg, (20, HEIGHT - 20))
        min_separation = self.minimum_separation()
        ms = self.font.render("Min. Separation: {}".format(round(min_separation, 1)), True, BLACK_RGB)
        self.screen.blit(ms, (20, HEIGHT - 40))
        v_closing = closing_speed(self.player, self.traffic[0])
        cs = self.font.render("Closing Speed: {}".format(round(v_closing, 1)), True, BLACK_RGB)
        self.screen.blit(cs, (20, HEIGHT - 60))
        d_closest = distance_closest_approach(self.player, self.traffic[0])
        dca = self.font.render("Closest approach: {}".format(round(d_closest, 1)), True, BLACK_RGB)
        self.screen.blit(dca, (20, HEIGHT - 80))

        # Display episode and 'time' (number of game loop iterations)
        st = self.font.render("Steps: {}".format(self.steps), True, BLACK_RGB)
        self.screen.blit(st, (WIDTH / 2 - 50, HEIGHT - 20))
        ep = self.font.render("Episode: {}".format(self.episode), True, BLACK_RGB)
        self.screen.blit(ep, (WIDTH / 2 - 50, HEIGHT - 40))

        # Display reward
        r_tot = self.font.render("Total reward: {}".format(round(self.total_reward, 1)), True, BLACK_RGB)
        self.screen.blit(r_tot, (WIDTH - 300, HEIGHT - 20))
        # d_goal = self.distance_to_goal()
        # r_dis = self.font.render("Step distance reward: {}".format(round(distance_reward(d_goal), 3)),
        #                          True, BLACK_RGB)
        # self.screen.blit(r_dis, (WIDTH - 200, HEIGHT - 60))
        psi = self.player.psi
        phi = relative_angle(self.player.x, self.player.y, self.goal_x, self.goal_y)
        r_psi = self.font.render("Step heading reward: {}".format(round(heading_reward(psi, phi), 3)),
                                 True, BLACK_RGB)
        self.screen.blit(r_psi, (WIDTH - 300, HEIGHT - 60))
        v_closing = closing_speed(self.player, self.traffic[0])
        r_clo = self.font.render("Step closing speed reward: {}".format(round(closing_speed_reward(v_closing), 3)),
                                 True, BLACK_RGB)
        self.screen.blit(r_clo, (WIDTH - 300, HEIGHT - 40))

        # Update the game screen
        pygame.display.update()
