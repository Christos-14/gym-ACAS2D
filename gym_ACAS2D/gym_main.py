import math
import gym
import numpy as np
from gym_ACAS2D.settings import *


def simulate():

    for episode in range(EPISODES):

        # Initialise the  environment
        environment = gym.make("ACAS2D-v0")
        state = environment.reset()

        total_reward = 0
        # AI tries up to MAX_TRY times
        for t in range(MAX_STEPS):
            # Quit if the game window closes
            if environment.quit:
                return -1
            # Fixed action selection for now
            # action = np.array([0])
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
    return 0


if __name__ == "__main__":
    simulate()
