import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_ACAS2D.envs.game import Game
import gym_ACAS2D.settings as settings


class ACAS2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        game = Game(settings.WIDTH, settings.HEIGHT,
                    settings.N_TRAFFIC, settings.AIRCRAFT_SIZE, settings.COLLISION_RADIUS, settings.MEDIUM_SPEED,
                    manual=False)

    def step(self, action):
        ...

    def reset(self):
        ...

    def render(self, mode='human'):
        ...

    def close(self):
        ...
