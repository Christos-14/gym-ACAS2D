import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_ACAS2D.envs.game import ACAS2DGame
import gym_ACAS2D.settings as settings


class ACAS2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.game = ACAS2DGame(settings.WIDTH, settings.HEIGHT,
                               settings.N_TRAFFIC, settings.AIRCRAFT_SIZE,
                               settings.COLLISION_RADIUS, settings.MEDIUM_SPEED,
                               manual=False)
        self.action_space = []
        self.observation_space = []

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
        self.game = self.game = ACAS2DGame(settings.WIDTH, settings.HEIGHT,
                                           settings.N_TRAFFIC, settings.AIRCRAFT_SIZE,
                                           settings.COLLISION_RADIUS, settings.MEDIUM_SPEED,
                                           manual=False)
        obs = self.game.observe()
        return obs

    def render(self, mode='human'):
        self.game.view()

    def close(self):
        ...
