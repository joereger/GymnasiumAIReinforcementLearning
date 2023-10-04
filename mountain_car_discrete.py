import gymnasium as gym
env = gym.make("MountainCar-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  
    observation, reward, terminated, truncated, info = env.step(action)
    print('observation: ', observation, 'reward: ', reward, 'terminated: ', terminated, 'truncated: ', truncated, 'info: ', info)

    if terminated or truncated:
        observation, info = env.reset()

env.close()