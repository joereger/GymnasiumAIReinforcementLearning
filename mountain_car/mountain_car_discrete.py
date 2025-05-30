import gymnasium as gym
import os

# Get project root directory (regardless of where script is executed from)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Constants
DATA_DIR = os.path.join(project_root, 'data', 'mountain_car')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

env = gym.make("MountainCar-v0", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  
    observation, reward, terminated, truncated, info = env.step(action)
    print('observation: ', observation, 'reward: ', reward, 'terminated: ', terminated, 'truncated: ', truncated, 'info: ', info)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# Note: This is a simple random agent implementation.
# When implementing learning algorithms, use DATA_DIR for saving models, e.g.:
# model_path = os.path.join(DATA_DIR, 'mountain_car_model.pkl')
