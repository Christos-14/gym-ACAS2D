from gym.envs.registration import register

register(
    id='ACAS2D-v0',
    entry_point='gym_ACAS2D.envs:ACAS2DEnv',
)
