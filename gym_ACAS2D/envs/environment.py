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
        # Keep track of whether the game window has been closed
        self.quit = False
        # Observation space: (x, y) position of the player, traffic aircraft and the goal.
        # Positions go from x=0 to x=WIDTH and from y=0 to y=HEIGHT
        pos_lo = np.zeros((N_TRAFFIC + 2) * 2, dtype=np.float32)
        pos_hi = np.array([WIDTH, HEIGHT] * (N_TRAFFIC + 2), dtype=np.float32)
        self.observation_space = spaces.Box(low=pos_lo, high=pos_hi, dtype=np.float32)
        # Action space: (lateral acceleration) combination set at time t
        action_lo = np.array([-ACC_LAT_LIMIT])
        action_hi = np.array([ACC_LAT_LIMIT])
        self.action_space = spaces.Box(low=action_lo, high=action_hi, dtype=np.float32)

    def step(self, action):
        self.game.action(action)
        obs = self.game.observe()
        reward = self.game.evaluate()
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
        self.quit = self.game.quit

    def close(self):
        ...
