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

    # Folders and file names
    best_model_save_path = "./models/best_model_{}".format(int(TOTAL_STEPS))
    best_model_file = best_model_save_path + "/best_model.zip"
    best_model_log_path = best_model_save_path + "/results"
    final_model_file = "./models/ACAS2D_PPO_{}.zip".format(int(TOTAL_STEPS))

    # Initialise the  environment
    environment = gym.make("ACAS2D-v0")

    # Check environment
    check_env(environment, warn=True, skip_render_check=True)

    # Train agent or load saved model
    try:
        model = PPO.load(best_model_file)
        print("Model loaded from file: {}".format(best_model_file))

    except FileNotFoundError:
        t_start = time.time()

        # Separate evaluation env
        eval_env = gym.make("ACAS2D-v0")
        eval_env = Monitor(eval_env)
        eval_callback = EvalCallback(eval_env,
                                     best_model_save_path=best_model_save_path,
                                     log_path=best_model_log_path,
                                     eval_freq=EVAL_STEPS,
                                     n_eval_episodes=10)
        # Create the callback list
        callback = CallbackList([eval_callback])

        model = PPO('MlpPolicy',
                    environment,
                    verbose=1,
                    seed=RANDOM_SEED,
                    tensorboard_log="./ACAS2D_PPO_tensorboard/")

        model.learn(total_timesteps=TOTAL_STEPS,
                    callback=callback,
                    tb_log_name="run_{}".format(TOTAL_STEPS))

        # Save final trained model
        model.save(final_model_file)
        print(f"Model training complete in {(time.time() - t_start) / 60.0} minutes.")

        # Load best model saved during training
        model = PPO.load(best_model_file)
        print("Model loaded from file: {}".format(best_model_file))

    # Test best trained model for a number of episodes
    for episode in range(1, EPISODES + 1):
        # # At the first episode, check the environment
        # if episode == 1:
        #     check_env(environment, warn=True, skip_render_check=True)
        # Reset the environment
        state = environment.reset()
        # Set game episode
        environment.game.episode = episode
        # # Episode reward
        # total_reward = 0
        # AI tries up to MAX_TRY times
        for t in range(MAX_STEPS):
            # Quit if the game window closes
            if environment.game.quit:
                exit()
            # The agent selects an action
            action, _states = model.predict(state)
            # Do action and get result
            next_state, reward, done, info = environment.step(action)
            # total_reward += reward
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
    log_to_file = False
    if log_to_file:
        orig_stdout = sys.stdout
        f = open('agent_log.txt', 'w')
        sys.stdout = f
    try:
        simulate(pause=False)
    except KeyboardInterrupt:
        pass
    if log_to_file:
        sys.stdout = orig_stdout
        f.close()
