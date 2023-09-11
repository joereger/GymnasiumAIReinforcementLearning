import gymnasium as gym
import random

env = gym.make("CartPole-v1", render_mode="human")

# Each of this episode is its own game.
for episode in range(10):
    env.reset()
    # this is each frame, up to 500...but we wont make it that far with random.
    for t in range(5000):
        # This will display the environment
        env.render()
        
        action = env.action_space.sample()

        # executes the environment with an action, 
        # and returns the observation of the environment, 
        # the reward, if the env is over, and other info.
        observation, reward, terminated, truncated, info = env.step(action)
        
        # print everything in one line:
        print(episode, t, observation, reward, terminated, truncated, info, action)
        if terminated:
            break
            