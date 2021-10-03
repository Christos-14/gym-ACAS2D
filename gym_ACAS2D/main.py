import pygame

from gym_ACAS2D.envs.game import ACAS2DGame
from gym_ACAS2D.settings import *


if __name__ == '__main__':

    # Initialize PyGame
    pygame.init()
    # Create the screen: WIDTH x HEIGHT
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    # Title and icon
    pygame.display.set_caption(CAPTION)
    pygame.display.set_icon(pygame.image.load(LOGO))
    # Load images
    playerIMG = pygame.image.load(PLAYER_IMG)
    goalIMG = pygame.image.load(GOAL_IMG)
    trafficIMG = pygame.image.load(TRAFFIC_IMG)
    # Text font
    font = pygame.font.Font(FONT_NAME, FONT_SIZE)

    # Create game
    game = ACAS2DGame(manual=True)

    # Game loop - keeps our screen active
    running = True
    time_steps = 0
    while running:
        # Count loop iterations (frames?)
        if game.running:
            time_steps += 1

        # Detect events
        for event in pygame.event.get():
            # Quit game
            if event.type == pygame.QUIT:
                running = False

        # Manual game play
        if game.manual:
            # Variables to track manual motion control inputs
            delta_x, delta_y = 0, 0
            playerStep = 2 * MEDIUM_SPEED
            # stores keys pressed
            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                delta_x = - playerStep
            if keys[pygame.K_RIGHT]:
                delta_x = playerStep
            if keys[pygame.K_UP]:
                delta_y = - playerStep
            if keys[pygame.K_DOWN]:
                delta_y = playerStep

        # Change background colour to sky colour RGB value
        screen.fill(SKY_RGB)

        if game.running:
            # Manual game play
            if game.manual:
                # Calculate new player position
                player_x = game.player.x + delta_x
                player_y = game.player.y + delta_y
                # Make sure the player stays on the screen
                if player_x < 0:
                    player_x = 0
                elif player_x + AIRCRAFT_SIZE > WIDTH:
                    player_x = WIDTH - AIRCRAFT_SIZE
                if player_y < 0:
                    player_y = 0
                elif player_y + AIRCRAFT_SIZE > HEIGHT:
                    player_y = HEIGHT - AIRCRAFT_SIZE
                # Update player position
                game.player.x = player_x
                game.player.y = player_y

            # Autonomous game play
            else:
                raise NotImplementedError

        # Place player in the game
        screen.blit(playerIMG, (player_x - (AIRCRAFT_SIZE/2), player_y - (AIRCRAFT_SIZE/2)))

        # Place goal in the game
        screen.blit(goalIMG, (game.goal_x - (AIRCRAFT_SIZE/2), game.goal_y - (AIRCRAFT_SIZE/2)))

        # Place traffic aircraft in the game
        for t in game.traffic:
            if game.running:
                t.update_position()
                if t.out_of_bounds(WIDTH, HEIGHT):
                    t.bounce(WIDTH, HEIGHT)
            screen.blit(trafficIMG, (t.x - (AIRCRAFT_SIZE/2), t.y - (AIRCRAFT_SIZE/2)))

        # Draw collision circle around aircraft
        pygame.draw.circle(screen, GREEN_RGB, (game.player.x, game.player.y), COLLISION_RADIUS, 1)
        # for t in game.traffic:
        #     pygame.draw.circle(screen, RED_RGB, (t.x, t.y), COLLISION_RADIUS, 1)

        # Display minimum separation
        min_separation = game.minimum_separation()
        ms = font.render("Min. Separation: {}".format(round(min_separation, 3)), True, FONT_RGB)
        screen.blit(ms, (20, HEIGHT-20))

        # Display 'time' (number of game loop iterations)
        ts = font.render("Time steps: {}".format(time_steps), True, FONT_RGB)
        screen.blit(ts, (round(WIDTH/2) - 50, HEIGHT - 20))

        # Display distance to target
        dist_to_goal = game.distance_to_goal()
        dg = font.render("Distance to goal: {}".format(round(dist_to_goal, 3)), True, FONT_RGB)
        screen.blit(dg, (WIDTH - 200, HEIGHT - 20))

        # Detect collisions
        if game.detect_collisions():
            mes = font.render("Collision!", True, FONT_RGB)
            screen.blit(mes, (round(WIDTH/2) - 30, round(HEIGHT/2)))

        # Check if player reached the goal
        if game.check_goal():
            mes = font.render("Goal reached!", True, FONT_RGB)
            screen.blit(mes, (round(WIDTH/2) - 40, round(HEIGHT/2)))

        # Update the game screen
        pygame.display.update()
