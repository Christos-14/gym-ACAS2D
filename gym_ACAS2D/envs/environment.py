import gym
import numpy as np
from gym.spaces import Dict, Box
from gym_ACAS2D.envs.game import ACAS2DGame
from gym_ACAS2D.settings import *


class ACAS2DEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):

        # Initialise game
        self.game = ACAS2DGame()

        # Observation space: (x, y, v_air, psi) for player, goal and traffic aircraft.
        obs_length = MAX_TRAFFIC + 2
        self.observation_space = Dict({
            "x": Box(low=0, high=WIDTH, shape=(obs_length,), dtype=np.float64),
            "y": Box(low=0, high=HEIGHT, shape=(obs_length,), dtype=np.float64),
            "v_air": Box(low=0,  # Goal speed is zero!
                         high=AIRSPEED_FACTOR_MAX*AIRSPEED,
                         shape=(obs_length,),
                         dtype=np.float64),
            "psi": Box(low=0, high=360, shape=(obs_length,), dtype=np.float64),
        })

        # Action space: (lateral acceleration) combination set at time t
        action_lo = np.array([-ACC_LAT_LIMIT])
        action_hi = np.array([ACC_LAT_LIMIT])
        self.action_space = Box(low=action_lo, high=action_hi, dtype=np.float64)

    def step(self, action):
        # Game clock tick
        self.game.clock.tick(FPS)
        # Take an action in the environment
        self.game.action(action)
        # Observe the environment
        obs = self.game.observe()
        # Retrieve reward
        reward = self.game.evaluate()
        # Check for termination
        done = self.game.is_done()
        # TODO: Add debugging info from the ACAS2DGame class
        info = {}
        return obs, reward, done, info

    def reset(self):
        del self.game
        self.game = ACAS2DGame()
        obs = self.game.observe()
        return obs

    def render(self, mode='human'):
        self.game.view()

    def close(self):
        ...
