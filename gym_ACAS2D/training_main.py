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


def training():

    # Initialise the  environment
    environment = gym.make("ACAS2D-v0")

    # Check environment
    check_env(environment, warn=True, skip_render_check=True)

    t_start = time.time()

    # Separate evaluation env
    eval_env = gym.make("ACAS2D-v0")
    eval_env = Monitor(eval_env)

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=best_model_save_path,
                                 log_path=best_model_log_path,
                                 eval_freq=EVAL_STEPS,
                                 n_eval_episodes=EVAL_EPISODES)
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


if __name__ == "__main__":

    log_file = "./models/logs/training_ACAS2D_PPO_{}_{}.txt".format(int(TOTAL_STEPS), MODEL_VERSION)
    log_to_file = True

    # Folders and file names
    best_model_save_path = "./models/best_model_{}_{}".format(int(TOTAL_STEPS), MODEL_VERSION)
    best_model_file = best_model_save_path + "/best_model.zip"
    best_model_log_path = best_model_save_path + "/results"
    final_model_file = "./models/ACAS2D_PPO_{}_{}.zip".format(int(TOTAL_STEPS), MODEL_VERSION)

    if log_to_file:
        orig_stdout = sys.stdout
        f = open(log_file, 'w')
        sys.stdout = f

    try:
        training()

    except KeyboardInterrupt:
        pass

    if log_to_file:
        sys.stdout = orig_stdout
        f.close()
