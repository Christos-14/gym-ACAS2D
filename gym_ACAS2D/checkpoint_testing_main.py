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
    e_d_path = []
    e_path = []
    e_t_paths = []
    # e_observations = []
    # e_actions = []
    # e_rewards = []

    e_heading_record = []
    e_d_sep_record = []
    e_a_lat_record = []

    e_d_goal_record = []
    e_delta_h_goal_record = []
    e_v_closing_record = []
    e_d_cpa_record = []
    e_d_dev_record = []

    e_step_reward_d_goal_record = []
    e_step_reward_h_goal_record = []
    e_step_reward_d_cpa_record = []
    e_step_reward_d_dev_record = []
    e_step_reward_record = []

    # Train agent or load saved model
    try:
        model = PPO.load(checkpoint_model_file)
        print("Model loaded from file: {}".format(checkpoint_model_file))

    except FileNotFoundError:
        print("Model could NOT be loaded from file: {}".format(checkpoint_model_file))

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
                # Log episode information
                e_outcome.append(OUTCOME_NAMES[environment.game.outcome])
                e_total_reward.append(environment.game.total_reward)
                e_time_steps.append(environment.game.steps)
                e_d_path.append(environment.game.d_path)
                e_path.append(environment.game.path)
                e_t_paths.append(environment.game.traffic_paths)
                # e_observations.append(environment.game.observations)
                # e_actions.append(environment.game.actions)
                # e_rewards.append(environment.game.rewards)
                e_heading_record.append(environment.game.heading_record)
                e_d_sep_record.append(environment.game.d_sep_record)
                e_a_lat_record.append(environment.game.a_lat_record)
                e_d_goal_record.append(environment.game.d_goal_record)
                e_delta_h_goal_record.append(environment.game.delta_h_goal_record)
                e_v_closing_record.append(environment.game.v_closing_record)
                e_d_cpa_record.append(environment.game.d_cpa_record)
                e_d_dev_record.append(environment.game.d_dev_record)
                e_step_reward_d_goal_record.append(environment.game.step_reward_d_goal_record)
                e_step_reward_h_goal_record.append(environment.game.step_reward_h_goal_record)
                e_step_reward_d_cpa_record.append(environment.game.step_reward_d_cpa_record)
                e_step_reward_d_dev_record.append(environment.game.step_reward_d_dev_record)
                e_step_reward_record.append(environment.game.step_reward_record)
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
    log_df["Path"] = e_path
    log_df["Traffic Paths"] = e_t_paths
    # log_df["Observations"] = e_observations
    # log_df["Actions"] = e_actions
    # log_df["Rewards"] = e_rewards
    log_df["psi"] = e_heading_record
    log_df["d_sep"] = e_d_sep_record
    log_df["a_lat"] = e_a_lat_record
    log_df["d_goal"] = e_d_goal_record
    log_df["delta_heading"] = e_delta_h_goal_record
    log_df["v_closing"] = e_v_closing_record
    log_df["d_cpa"] = e_d_cpa_record
    log_df["d_dev"] = e_d_dev_record
    log_df["r_d_goal"] = e_step_reward_d_goal_record
    log_df["r_h_goal"] = e_step_reward_h_goal_record
    log_df["r_d_cpa"] = e_step_reward_d_cpa_record
    log_df["r_d_dev"] = e_step_reward_d_dev_record
    log_df["r_step"] = e_step_reward_record
    log_df.to_csv(test_data_file, index=False)


if __name__ == "__main__":

    checkpoint_time_steps = int(EVAL_STEPS * 16)

    log_file = "./models/logs/checkpoint_testing_ACAS2D_PPO_{}_{}_at_{}.txt".format(int(TOTAL_STEPS),
                                                                                    MODEL_VERSION,
                                                                                    checkpoint_time_steps)
    log_to_file = False

    # Model files
    checkpoints_model_save_path = "./models/checkpoints_{}_{}".format(int(TOTAL_STEPS), MODEL_VERSION)
    checkpoint_model_file = checkpoints_model_save_path + "/model_{}_steps.zip".format(checkpoint_time_steps)

    # Testing data file
    test_data_file = "./models/logs/checkpoint_testing_ACAS2D_PPO_{}_{}_at_{}_{}.csv".format(int(TOTAL_STEPS),
                                                                                             MODEL_VERSION,
                                                                                             checkpoint_time_steps,
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
