import gymnasium as gym
import os

# Get project root directory (regardless of where script is executed from)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Constants
DATA_DIR = os.path.join(project_root, 'data', 'lunar_lander')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()

# Note: This is a simple random agent implementation.
# When implementing learning algorithms, use DATA_DIR for saving models, e.g.:
# model_path = os.path.join(DATA_DIR, 'lunar_lander_model.pkl')
