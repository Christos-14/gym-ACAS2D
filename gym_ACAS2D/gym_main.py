from gym_ACAS2D.settings import *

from stable_baselines3.common.env_checker import check_env

import math
import gym
import random
import numpy as np


# Initialise random generator
random.seed(RANDOM_SEED)


def simulate(pause=False):

    for episode in range(1, EPISODES+1):
        # Initialise the  environment
        environment = gym.make("ACAS2D-v0")
        # At the first episode, check the environment
        if episode == 1:
            check_env(environment, warn=True, skip_render_check=True)
        # Reset the environment
        state = environment.reset()
        # Set game episode
        environment.game.episode = episode
        # Episode reward
        total_reward = 0
        # AI tries up to MAX_TRY times
        for t in range(MAX_STEPS):
            # Quit if the game window closes
            if environment.game.quit:
                return -1
            # Fixed action selection for now
            if episode % 2 == 0:
                action = np.array([0])
            else:
                action = np.array([-ACC_LAT_LIMIT * np.sin(2 * math.pi * (t / FPS))])
            # Do action and get result
            next_state, reward, done, info = environment.step(action)
            total_reward += reward
            # Set up for the next iteration
            state = next_state
            # Draw games
            environment.render()
            # When episode is done, print reward
            if done:
                print("Episode {:<3}: Time steps: {:<7} - Outcome: {:<10} - Total Reward = {}"
                      .format(episode, t, OUTCOME_NAMES[environment.game.outcome], total_reward))
                break
            # Pause the game screen to start video capture
            if pause and episode == 0 and t == 0:
                input("Press any key to continue")
    return 0


if __name__ == "__main__":
    simulate(pause=False)

