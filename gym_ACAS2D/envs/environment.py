import gym
import numpy as np
from gym import spaces
from gym_ACAS2D.envs.game import ACAS2DGame
from gym_ACAS2D.settings import *


class ACAS2DEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):

        # Initialise game
        self.game = ACAS2DGame()

        # Observation space: (x, y, v_air, psi) for player, goal and traffic aircraft.
        lo = np.array([[0]*(MAX_TRAFFIC+2),
                       [0]*(MAX_TRAFFIC+2),
                       [AIRSPEED_FACTOR_MIN * AIRSPEED] * (MAX_TRAFFIC+2),
                       [0]*(MAX_TRAFFIC+2)])
        hi = np.array([[WIDTH]*(MAX_TRAFFIC+2),
                       [HEIGHT]*(MAX_TRAFFIC+2),
                       [AIRSPEED_FACTOR_MAX * AIRSPEED] * (MAX_TRAFFIC+2),
                       [360]*(MAX_TRAFFIC+2)])
        self.observation_space = spaces.Box(low=lo, high=hi, dtype=np.float32)

        # Action space: (lateral acceleration) combination set at time t
        action_lo = np.array([-ACC_LAT_LIMIT])
        action_hi = np.array([ACC_LAT_LIMIT])
        self.action_space = spaces.Box(low=action_lo, high=action_hi, dtype=np.float32)

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
