from gym_ACAS2D.settings import *

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

import random
import time
import gym
import sys

# Initialise random generator
random.seed(RANDOM_SEED)


def simulate(pause=False):

    # Initialise the  environment
    environment = gym.make("ACAS2D-v0")

    # Check environment
    check_env(environment, warn=True, skip_render_check=True)

    # Train agent or load saved model
    try:
        # best_model_save_path = "./models/best_model_{}_4".format(int(TOTAL_STEPS))
        # best_model_file = best_model_save_path + "/best_model.zip"
        model = PPO.load(best_model_file)
        print("Model loaded from file: {}".format(best_model_file))

    except FileNotFoundError:
        print("Model could NOT be loaded from file: {}".format(best_model_file))

    # Test best trained model for a number of episodes
    for episode in range(1, TEST_EPISODES + 1):
        # Reset the environment
        state = environment.reset()
        # Set game episode
        environment.game.episode = episode

        # Agent has MAX_STEPS to win the episode.
        for t in range(MAX_STEPS):
            # Quit if the game window closes
            if environment.game.quit:
                exit()
            # The agent selects an action
            action, _states = model.predict(state, deterministic=True)
            # Do action and get result
            next_state, reward, done, info = environment.step(action)
            # Set up for the next iteration
            state = next_state
            # Draw games
            environment.render()
            # When episode is done, print reward
            if done:
                print("Episode {:<3}: Time steps: {:<7} - Outcome: {:<10} - Total Reward = {}"
                      .format(episode, t, OUTCOME_NAMES[environment.game.outcome], environment.game.total_reward))
                break
            # Pause the game screen to start video capture
            if pause and episode == 0 and t == 0:
                input("Press any key to continue")


if __name__ == "__main__":

    total_steps = 2048 * 512
    version = 4

    log_file = "./models/logs/testing_ACAS2D_PPO_{}_{}.txt".format(int(total_steps), version)
    log_to_file = True

    # Folders and file names
    best_model_save_path = "./models/best_model_{}_{}".format(int(total_steps), version)
    best_model_file = best_model_save_path + "/best_model.zip"

    if log_to_file:
        orig_stdout = sys.stdout
        f = open(log_file, 'w')
        sys.stdout = f
    try:
        simulate(pause=False)
    except KeyboardInterrupt:
        pass
    if log_to_file:
        sys.stdout = orig_stdout
        f.close()
