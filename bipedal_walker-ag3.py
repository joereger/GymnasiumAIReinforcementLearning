import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import pygame
import os

class AG3Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(AG3Network, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)
        
        init.normal_(self.fc1.weight, mean=0., std=1)
        init.normal_(self.fc1.bias, mean=0., std=1)
        init.normal_(self.fc2.weight, mean=0., std=1)
        init.normal_(self.fc2.bias, mean=0., std=1)
        init.normal_(self.fc3.weight, mean=0., std=1)
        init.normal_(self.fc3.bias, mean=0., std=1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x




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

