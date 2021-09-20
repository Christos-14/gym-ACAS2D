import numpy as np
import random
import pygame

from gym_ACAS2D.envs.aircraft import PlayerAircraft, TrafficAircraft
import gym_ACAS2D.settings as settings


class ACAS2DGame:
    def __init__(self, static=False, manual=False):

        # Initialize PyGame
        pygame.init()
        # Title and icon
        pygame.display.set_caption(settings.CAPTION)
        pygame.display.set_icon(pygame.image.load(settings.LOGO))

        # Create the screen: WIDTH x HEIGHT
        self.screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
        # # Game clock
        # self.clock = pygame.time.Clock()
        self.steps = 0

        # Flags
        self.running = True    # Is the game running?
        self.win = None    # Did the player win or lose?
        self.manual = manual  # Is the player controlled manually?

        # ACAS2DGame mode: True for static, False for random
        self.static = static

        # Load images
        self.playerIMG = pygame.image.load(settings.PLAYER_IMG)
        self.goalIMG = pygame.image.load(settings.GOAL_IMG)
        self.trafficIMG = pygame.image.load(settings.TRAFFIC_IMG)
        # Text font
        self.font = pygame.font.Font(settings.FONT_NAME, settings.FONT_SIZE)

        if not static:
            # Set the player starting position at random, in the bottom part of the airspace
            # Set the player starting heading and speed to zero
            player_x = random.randint(0, settings.WIDTH - settings.AIRCRAFT_SIZE)
            player_y = random.randint(round(4 * settings.HEIGHT / 5), settings.HEIGHT - settings.AIRCRAFT_SIZE)
            self.player = PlayerAircraft(x=player_x, y=player_y, speed=0, heading=0)

            # Set the traffic aircraft positions, headings and speeds at random, in the middle part of the airspace.
            self.traffic = []
            for t in range(settings.N_TRAFFIC):
                # Random position in the mid part of the airspace
                t_x = random.randint(0, settings.WIDTH - settings.AIRCRAFT_SIZE)
                t_y = random.randint(0, round(3 * settings.HEIGHT / 5))
                # Random speed: low (75%), medium (100%), or high (125%)
                t_speed = (random.randint(3, 6) / 4) * settings.MEDIUM_SPEED
                # Random heading: 0..359 degrees
                t_heading = random.randint(0, 360)
                self.traffic.append(TrafficAircraft(x=t_x, y=t_y, speed=t_speed, heading=t_heading))

            # Set the goal position at random, in the top part of the airspace.
            self.goal_x = random.randint(settings.AIRCRAFT_SIZE, settings.WIDTH - settings.AIRCRAFT_SIZE)
            self.goal_y = random.randint(settings.AIRCRAFT_SIZE, round(settings.HEIGHT / 5))

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
            if d < settings.COLLISION_RADIUS:
                collision = True
                self.running = False
                self.win = False
                break
        return collision

    def check_goal(self, d_reached=20):
        # Player and goal positions as np.array
        p = np.array((self.player.x, self.player.y))
        g = np.array((self.goal_x, self.goal_y))
        # Euclidean distance between player and goal
        d = np.linalg.norm(p-g, 2)
        # If the distance is less than the collision radius, player reached the goal
        goal = d < d_reached
        if goal:
            self.running = False
            self.win = True
        return goal

    def observe(self):
        # Increase number of steps in the game (all steps start with an observation)
        self.steps += 1
        # We create the dictionary of observations, as expected by the environment's observation_space
        obs = {}
        # Player position, speed and heading
        pos = [[self.player.x, self.player.y]]
        spd = [self.player.speed]
        hed = [self.player.heading]
        # Traffic positions, speeds and headings
        for t in self.traffic:
            pos.append([t.x, t.y])
            spd.append(t.speed)
            hed.append(t.heading)
        # Goal position
        pos.append([self.goal_x, self.goal_y])

        obs["position"] = np.array(pos)
        obs["speed"] = np.array(spd)
        obs["heading"] = np.array(hed)

        return obs

    def action(self, action):
        # Update player speed and heading based on action taken
        self.player.speed = action["speed"]
        self.player.heading = action["heading"]
        # Update player position based on that speed and heading
        self.player.update_position()
        # If the game is still running, update the traffic aircraft positions.
        for t in self.traffic:
            if self.running:
                t.update_position()
                if t.out_of_bounds(settings.WIDTH, settings.HEIGHT):
                    t.bounce(settings.WIDTH, settings.HEIGHT)

    def evaluate(self):
        raise NotImplementedError

    def is_done(self):
        # The game ends either when the player has reached the goal (win) or when there's a collision (lose)
        raise self.check_goal() or self.detect_collisions()

    def view(self):
        # Detect events
        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                self.running = False

        # Change background colour to sky colour RGB value
        self.screen.fill(settings.SKY_RGB)

        # Place player in the game
        self.screen.blit(self.playerIMG, (self.player.x - (settings.AIRCRAFT_SIZE / 2),
                                          self.player.y - (settings.AIRCRAFT_SIZE / 2)))

        # Place goal in the game
        self.screen.blit(self.goalIMG, (self.goal_x - (settings.AIRCRAFT_SIZE / 2),
                                        self.goal_y - (settings.AIRCRAFT_SIZE / 2)))

        # Place traffic aircraft in the game
        for t in self.traffic:
            self.screen.blit(self.trafficIMG, (t.x - (settings.AIRCRAFT_SIZE / 2),
                                               t.y - (settings.AIRCRAFT_SIZE / 2)))

        # Draw collision circle around aircraft
        pygame.draw.circle(self.screen, settings.GREEN_RGB, (self.player.x, self.player.y),
                           settings.COLLISION_RADIUS, 1)
        # for t in self.traffic:
        #     pygame.draw.circle(self.screen, settings.RED_RGB, (t.x, t.y), settings.COLLISION_RADIUS, 1)

        # Display minimum separation
        min_separation = self.minimum_separation()
        ms = self.font.render("Min. Separation: {}".format(round(min_separation, 3)), True, settings.FONT_RGB)
        self.screen.blit(ms, (20, settings.HEIGHT - 20))

        # Display 'time' (number of game loop iterations)
        ts = self.font.render("Time steps: {}".format(self.time), True, settings.FONT_RGB)
        self.screen.blit(ts, (round(settings.WIDTH / 2) - 50, settings.HEIGHT - 20))

        # Display distance to target
        dist_to_goal = self.distance_to_goal()
        dg = self.font.render("Distance to goal: {}".format(round(dist_to_goal, 3)), True, settings.FONT_RGB)
        self.screen.blit(dg, (settings.WIDTH - 200, settings.HEIGHT - 20))

        # Detect collisions
        if self.detect_collisions():
            mes = self.font.render("Collision!", True, settings.FONT_RGB)
            self.screen.blit(mes, (round(settings.WIDTH / 2) - 30, round(settings.HEIGHT / 2)))

        # Check if player reached the goal
        if self.check_goal():
            mes = self.font.render("Goal reached!", True, settings.FONT_RGB)
            self.screen.blit(mes, (round(settings.WIDTH / 2) - 40, round(settings.HEIGHT / 2)))

        # Update the game screen
        pygame.display.update()
