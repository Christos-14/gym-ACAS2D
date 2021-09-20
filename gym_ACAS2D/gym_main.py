import gym
import random
import gym_ACAS2D.settings as settings


def simulate():

    for episode in range(settings.EPISODES):

        # Init environment
        environment = gym.make("ACAS2D-v0")
        state = environment.reset()
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(settings.MAX_STEPS):

            # Fixed action selection for now
            action = {"speed": settings.MEDIUM_SPEED,
                      "heading": random.uniform(0, 1) * 360}

            # Do action and get result
            next_state, reward, done, info = environment.step(action)
            total_reward += reward

            # Set up for the next iteration
            state = next_state

            # Draw games
            environment.render()

            # When episode is done, print reward
            if done:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                break


if __name__ == "__main__":
    simulate()
