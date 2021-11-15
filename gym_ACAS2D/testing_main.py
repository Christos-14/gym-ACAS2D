from gym_ACAS2D.settings import *

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

import pandas as pd

import random
import gym
import sys

# Initialise random generator
random.seed(RANDOM_SEED)


def simulate(pause=False):

    # Initialise the  environment
    environment = gym.make("ACAS2D-v0")

    # Check environment
    check_env(environment, warn=True, skip_render_check=True)

    # Episode metrics
    e_outcome = []
    e_total_reward = []
    e_time_steps = []
    e_path = []
    e_t_paths = []
    e_d_path = []
    e_d_cpa = []
    e_d_goal_final = []

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
                # Log episode information
                e_outcome.append(OUTCOME_NAMES[environment.game.outcome])
                e_total_reward.append(environment.game.total_reward)
                e_time_steps.append(environment.game.steps)
                e_d_path.append(environment.game.d_path)
                e_path.append(environment.game.path)
                e_t_paths.append(environment.game.traffic_paths)
                e_d_cpa.append(environment.game.d_closest_approach)
                e_d_goal_final.append(environment.game.distance_to_goal())
                print("Episode {:<3}: Time steps: {:<7} - Outcome: {:<10} - Total Reward = {}"
                      .format(episode, t, OUTCOME_NAMES[environment.game.outcome], environment.game.total_reward))
                break
            # Pause the game screen to start video capture
            if pause and episode == 0 and t == 0:
                input("Press any key to continue")

    # Log testing data
    log_df = pd.DataFrame()
    log_df["Episode"] = range(1, TEST_EPISODES + 1)
    log_df["Outcome"] = e_outcome
    log_df["Total Reward"] = e_total_reward
    log_df["Time Steps"] = e_time_steps
    log_df["Path Length"] = e_d_path
    log_df["Closest Approach"] = e_d_cpa
    log_df["Final Distance to Goal"] = e_d_goal_final
    log_df["Path"] = e_path
    log_df["Traffic Paths"] = e_t_paths
    log_df.to_csv(test_data_file, index=False)


if __name__ == "__main__":

    log_file = "./models/logs/testing_ACAS2D_PPO_{}_{}.txt".format(int(TOTAL_STEPS), MODEL_VERSION)
    log_to_file = False

    # Model files
    best_model_save_path = "./models/best_model_{}_{}".format(int(TOTAL_STEPS), MODEL_VERSION)
    best_model_file = best_model_save_path + "/best_model.zip"

    # Testing data file
    test_data_file = "./models/logs/testing_ACAS2D_PPO_{}_{}_{}.csv".format(int(TOTAL_STEPS),
                                                                            MODEL_VERSION,
                                                                            TEST_EPISODES)

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
