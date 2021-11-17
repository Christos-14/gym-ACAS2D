import random
import pygame

from gym_ACAS2D.envs.aircraft import PlayerAircraft, TrafficAircraft
from gym_ACAS2D.envs.rewards import *


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
        # Distance covered by the player
        self.d_path = 0
        # Player's path
        self.path = []

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

        # Set the goal position
        self.goal_x = WIDTH - GOAL_RADIUS
        self.goal_y = HEIGHT / 2

        # Set the player starting position at random, in the bottom part of the airspace
        # Set the player starting psi and v_air
        player_x = COLLISION_RADIUS
        player_y = HEIGHT / 2
        player_speed = AIRSPEED
        player_psi = random.uniform(0, 360)
        self.player = PlayerAircraft(x=player_x, y=player_y, v_air=player_speed, psi=player_psi)
        # Set player's initial heading towards the general direction of the goal.
        self.player.psi = (relative_angle(self.player.x, self.player.y, self.goal_x, self.goal_y) +
                           random.uniform(-PLAYER_INITIAL_HEADING_LIM, PLAYER_INITIAL_HEADING_LIM)) % 360

        # Initialise player's path record with initial position
        self.path.append((self.player.x, self.player.y))
        # Initial distance to goal
        self.d_goal_initial = self.distance_to_goal()
        # Maximum possible distance from goal
        self.d_goal_max = self.distance_to_goal() + (AIRSPEED / FPS) * MAX_STEPS
        # Maximum possible deviation from straight line plan
        self.d_dev_max = (AIRSPEED / FPS) * MAX_STEPS
        # Maximum possible separation
        self.d_separation_max = np.sqrt(WIDTH ** 2 + HEIGHT ** 2) + (2 * (AIRSPEED / FPS) * MAX_STEPS)
        # Maximum distance of closest approach
        self.d_cpa_max = np.sqrt(WIDTH ** 2 + HEIGHT ** 2)
        # Maximum (absolute) closing speed
        self.v_closing_max = 2 * (AIRSPEED_FACTOR_MAX * AIRSPEED)

        # Number of traffic aircraft
        self.num_traffic = random.randint(MIN_TRAFFIC, MAX_TRAFFIC)
        # Set the traffic aircraft positions, headings and speeds at random, in the middle part of the airspace.
        self.traffic = []
        # Traffic paths
        self.traffic_paths = [[] for _ in range(self.num_traffic)]
        for n in range(self.num_traffic):

            if n == 0:
                starts_down = random.randint(0, 1)
                # Random position in the mid part of the airspace
                t_x = WIDTH - COLLISION_RADIUS
                t_y = COLLISION_RADIUS + (starts_down * (HEIGHT - (2 * COLLISION_RADIUS)))
                # Random v_air
                t_speed = random.uniform(AIRSPEED_FACTOR_MIN, AIRSPEED_FACTOR_MAX) * AIRSPEED
                # Random psi: 0..360 degrees
                t_heading = (135 + (starts_down * 90) +
                             random.uniform(-TRAFFIC_INITIAL_HEADING_LIM, TRAFFIC_INITIAL_HEADING_LIM)) % 360

            else:
                # Random position in the mid part of the airspace
                t_x = random.uniform(0, WIDTH - AIRCRAFT_SIZE)
                t_y = random.uniform(0, 3 * HEIGHT / 5)
                # Random v_air
                t_speed = random.uniform(AIRSPEED_FACTOR_MIN, AIRSPEED_FACTOR_MAX) * AIRSPEED
                # Random psi: 0..360 degrees
                t_heading = random.uniform(0, 360)

            self.traffic.append(TrafficAircraft(x=t_x, y=t_y, v_air=t_speed, psi=t_heading))
            # Initialise traffic aircraft's path record with initial position
            self.traffic_paths[n].append((t_x, t_y))

        # Player's closest approach to traffic throughout the episode
        self.d_closest_approach = self.minimum_separation()

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

    def plan_deviation(self):
        # The deviation (distance) from the straight line trajectory to the goal
        d_goal = self.distance_to_goal()
        h_goal = self.heading_to_goal()
        h_goal_rad = (h_goal / 360.0) * 2 * math.pi
        return d_goal * np.sin(h_goal_rad)

    def check_timeout(self):
        return self.steps > MAX_STEPS

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
               self.player.psi / 360,
               self.plan_deviation() / self.d_dev_max,
               self.distance_to_goal() / self.d_goal_max,
               self.heading_to_goal() / 360]

        for t in self.traffic:
            x1, y1 = self.player.x, self.player.y
            x2, y2 = t.x, t.y
            obs.append(distance(x1, y1, x2, y2) / self.d_separation_max)
            obs.append(distance_closest_approach(self.player, t) / self.d_cpa_max)
            obs.append(closing_speed(self.player, t) / self.v_closing_max)

        # Padding
        obs += [0] * (2 * (MAX_TRAFFIC - self.num_traffic))

        obs = np.array(obs).astype(np.float64)

        return obs

    def action(self, action):
        # Update player a_lat based on action taken
        # Action is scaled to [-1, 1] ; scale to original [-ACC_LAT_LIMIT, ACC_LAT_LIMIT]
        self.player.a_lat = action[0] * ACC_LAT_LIMIT
        # Keep current position to calculate distance covered
        x_old, y_old = self.player.x, self.player.y
        # Update player position based on that v_air and psi
        self.player.update_state()
        # Update path records
        self.path.append((self.player.x, self.player.y))
        for n in range(self.num_traffic):
            self.traffic_paths[n].append((self.traffic[n].x, self.traffic[n].y))
        # Update path distance record
        self.d_path += distance(x_old, y_old, self.player.x, self.player.y)
        # Check and update closest approach record
        if self.minimum_separation() < self.d_closest_approach:
            self.d_closest_approach = self.minimum_separation()
        # If the game is still running, update the traffic aircraft positions.
        for t in self.traffic:
            if self.running:
                t.update_state()

    def evaluate(self):

        # Step reward
        psi = self.player.psi
        phi = relative_angle(self.player.x, self.player.y, self.goal_x, self.goal_y)
        v_closing = closing_speed(self.player, self.traffic[0])
        d_cpa = distance_closest_approach(self.player, self.traffic[0])
        d_goal = self.distance_to_goal()
        d_dev = self.plan_deviation()

        r_step = step_reward_6(v_closing, psi, phi, d_cpa, d_goal, d_dev)

        # Time discount factor
        tdf = 1 - (self.steps / MAX_STEPS)
        reward = r_step * tdf

        # Penalise collisions.
        if self.detect_collisions():
            reward += REWARD_COLLISION

        # Reward reaching the goal
        if self.check_goal():
            reward += REWARD_GOAL

        # Accumulate episode rewards
        self.total_reward += reward

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
        # if done:
        #     print("is_done() 	>>> Outcome: {:<10} - Total Reward: {}".
        #           format(OUTCOME_NAMES[self.outcome].upper(), self.total_reward))
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

        # Draw goal radius around goal
        pygame.draw.circle(self.screen, YELLOW_RBG, (self.goal_x, self.goal_y),
                           GOAL_RADIUS, 1)

        # Draw collision radius around traffic aircraft
        for t in self.traffic:
            pygame.draw.circle(self.screen, RED_RGB, (t.x, t.y), COLLISION_RADIUS, 1)

        # Display player's state
        pos = self.font.render("pos: ({}, {})".format(round(self.player.x, 1),
                                                           round(self.player.y, 1)), True, BLACK_RGB)
        self.screen.blit(pos, (20, 20))
        vair = self.font.render("v_air: {}".format(round(self.player.v_air, 1)), True, BLACK_RGB)
        self.screen.blit(vair, (20, 40))
        psi = self.font.render("psi: {}".format(round(self.player.psi, 1)), True, BLACK_RGB)
        self.screen.blit(psi, (20, 60))
        psi_dot = self.font.render("psi_dot: {}".format(round(self.player.psi_dot, 1)), True, BLACK_RGB)
        self.screen.blit(psi_dot, (20, 80))
        alat = self.font.render("a_lat: {}".format(round(self.player.a_lat, 1)), True, BLACK_RGB)
        self.screen.blit(alat, (20, 100))
        n_alat = self.font.render("a_lat_norm: {}".format(round(self.player.a_lat / ACC_LAT_LIMIT, 3)),
                                     True, BLACK_RGB)
        self.screen.blit(n_alat, (20, 120))

        # Display metrics
        d_goal = self.distance_to_goal()
        dg = self.font.render("Distance to goal: {}".format(round(d_goal, 1)), True, BLACK_RGB)
        self.screen.blit(dg, (20, HEIGHT - 20))
        min_separation = self.minimum_separation()
        ms = self.font.render("Min. Separation: {}".format(round(min_separation, 1)), True, BLACK_RGB)
        self.screen.blit(ms, (20, HEIGHT - 40))
        rel_angle_to_traffic = relative_angle(self.player.x, self.player.y,
                                              self.traffic[0].x, self.traffic[0].y)
        ratt = self.font.render("Rel. angle to traffic: {}".format(round(rel_angle_to_traffic, 1)), True, BLACK_RGB)
        self.screen.blit(ratt, (20, HEIGHT - 60))
        v_closing = closing_speed(self.player, self.traffic[0])
        cs = self.font.render("Closing Speed: {}".format(round(v_closing, 1)), True, BLACK_RGB)
        self.screen.blit(cs, (20, HEIGHT - 80))
        d_closest = distance_closest_approach(self.player, self.traffic[0])
        dca = self.font.render("Closest approach: {}".format(round(d_closest, 1)), True, BLACK_RGB)
        self.screen.blit(dca, (20, HEIGHT - 100))
        hg = self.font.render("Delta heading : {}".format(round(delta_heading(self.player.psi,
                                                                              self.heading_to_goal()), 1)),
                              True, BLACK_RGB)
        self.screen.blit(hg, (20, HEIGHT - 120))
        d_dev = self.plan_deviation()
        dev = self.font.render("Plan deviation : {}".format(round(d_dev, 1)), True, BLACK_RGB)
        self.screen.blit(dev, (20, HEIGHT - 140))

        # Display episode and 'time' (number of game loop iterations)
        st = self.font.render("Steps: {}".format(self.steps), True, BLACK_RGB)
        self.screen.blit(st, (WIDTH / 2 - 50, HEIGHT - 20))
        ep = self.font.render("Episode: {}".format(self.episode), True, BLACK_RGB)
        self.screen.blit(ep, (WIDTH / 2 - 50, HEIGHT - 40))

        # Display reward
        r_tot = self.font.render("Total reward: {}".format(round(self.total_reward, 1)), True, BLACK_RGB)
        self.screen.blit(r_tot, (WIDTH - 300, HEIGHT - 20))
        psi = self.player.psi
        phi = relative_angle(self.player.x, self.player.y, self.goal_x, self.goal_y)
        r_psi = self.font.render("Step heading reward: {}".
                                 format(round(heading_reward(psi, phi), 3)),
                                 True, BLACK_RGB)
        self.screen.blit(r_psi, (WIDTH - 300, HEIGHT - 120))
        v_closing = closing_speed(self.player, self.traffic[0])
        d_cpa = distance_closest_approach(self.player, self.traffic[0])
        r_cpa = self.font.render("Step closest approach reward: {}".
                                 format(round(closest_approach_reward(v_closing, d_cpa), 3)),
                                 True, BLACK_RGB)
        self.screen.blit(r_cpa, (WIDTH - 300, HEIGHT - 100))
        d_goal = self.distance_to_goal()
        r_d = self.font.render("Step goal distance reward: {}".
                                format(round(distance_reward(d_goal), 3)),
                                True, BLACK_RGB)
        self.screen.blit(r_d, (WIDTH - 300, HEIGHT - 80))
        d_dev = self.plan_deviation()
        r_dev = self.font.render("Step plan deviation reward: {}".format(round(plan_deviation_reward(d_dev), 3)),
                                 True, BLACK_RGB)
        self.screen.blit(r_dev, (WIDTH - 300, HEIGHT - 60))
        r_step = self.font.render("Step reward: {}".format(round(step_reward_6(v_closing,
                                                                               self.player.psi,
                                                                               self.heading_to_goal(),
                                                                               d_cpa,
                                                                               d_goal,
                                                                               d_dev), 3)),
                                  True, BLACK_RGB)
        self.screen.blit(r_step, (WIDTH - 300, HEIGHT - 40))

        # Update the game screen
        pygame.display.update()
