import gym
import numpy as np
from gym import spaces
from gym_ACAS2D.envs.game import ACAS2DGame
import gym_ACAS2D.settings as settings


class ACAS2DEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        # Initialise game
        self.game = ACAS2DGame()
        # Keep track of whether the game window has been closed
        self.quit = False
        # Observation space: (x, y, v, theta) state for the player and traffic aircraft and the goal position.
        # Positions go from x=0 to x=WIDTH and from y=0 to y=HEIGHT
        pos_lo = np.zeros((settings.N_TRAFFIC + 2) * 2, dtype=np.float32)
        pos_hi = np.array([settings.WIDTH, settings.HEIGHT] * (settings.N_TRAFFIC + 2), dtype=np.float32)
        # Speeds go from v=0 to v=1.25*SPEED_MEDIUM
        speed_lo = np.ones((settings.N_TRAFFIC + 1), dtype=np.float32) * (settings.MEDIUM_SPEED * settings.MIN_SPEED_FACTOR)
        speed_hi = np.ones((settings.N_TRAFFIC + 1), dtype=np.float32) * (settings.MEDIUM_SPEED * settings.MAX_SPEED_FACTOR)
        # Headings go from theta=0 to theta = 360
        head_lo = np.zeros((settings.N_TRAFFIC + 1), dtype=np.float32)
        head_hi = np.ones((settings.N_TRAFFIC + 1), dtype=np.float32) * 360
        self.observation_space = spaces.Dict({"position": spaces.Box(low=pos_lo, high=pos_hi, dtype=np.float32),
                                              "speed": spaces.Box(low=speed_lo, high=speed_hi, dtype=np.float32),
                                              "heading": spaces.Box(low=head_lo, high=head_hi, dtype=np.float32)})
        # Action space: (v, theta) combination set at time t
        action_lo = np.array([(settings.MEDIUM_SPEED * settings.MIN_SPEED_FACTOR), 0])
        action_hi = np.array([(settings.MEDIUM_SPEED * settings.MAX_SPEED_FACTOR), 360])
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
