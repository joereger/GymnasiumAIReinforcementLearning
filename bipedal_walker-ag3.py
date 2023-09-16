import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import pygame
import os





def evaluate(agent, env, num_episodes=10):
    episode_rewards = []
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        while True:
            action = agent.actor(torch.FloatTensor(state)).detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = next_state
            if terminated or truncated:
                break
        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {i_episode+1}: Reward = {episode_reward}")
    avg_reward = sum(episode_rewards) / num_episodes
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

# Set up the environment
#env = gym.make("BipedalWalker-v3", render_mode="human")
env = gym.make("BipedalWalker-v3")

# Constants
num_episodes = 1000
PATH = 'data/'
PREFIX = 'bipedal_walker_v04'

# Initialize the PPOAgent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPOAgent(state_dim, action_dim)


# Load the agents
agent.load(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')

# Train the agent
agent.train(env, num_episodes)

# Save the agents
agent.save(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')

# Evaluate the training
#evaluate(agent, env)

