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
        pos_lo = np.zeros((settings.N_TRAFFIC+2, 2))
        pos_hi = np.ones((settings.N_TRAFFIC+2, 2))
        pos_hi[:, 0] *= settings.WIDTH
        pos_hi[:, 1] *= settings.HEIGHT
        # Speeds go from v=0 to v=1.25*SPEED_MEDIUM
        speed_min = np.zeros((settings.N_TRAFFIC+1, 1))
        speed_max = np.ones((settings.N_TRAFFIC+1, 1)) * (settings.MEDIUM_SPEED * settings.MAX_SPEED_FACTOR)
        # Headings go from theta=0 to theta = 360
        head_min = np.zeros((settings.N_TRAFFIC+1, 1))
        head_max = np.ones((settings.N_TRAFFIC+1, 1))*360
        self.observation_space = spaces.Dict({"position": spaces.Box(low=pos_lo, high=pos_hi, dtype=np.float32),
                                              "speed": spaces.Box(low=speed_min, high=speed_max, dtype=np.float32),
                                              "heading": spaces.Box(low=head_min, high=head_max, dtype=np.float32)})
        # Action space: (v, theta) combination set at time t
        self.action_space = spaces.Dict({"speed": spaces.Box(low=speed_min, high=speed_max, dtype=np.float32),
                                        "heading": spaces.Box(low=head_min, high=head_max, dtype=np.float32)})

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
