import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import pygame
import os

class A3CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3CNetwork, self).__init__()
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

class A3CAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0001):
        self.ag3_network = A3CNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ag3_network.parameters(), lr=learning_rate)

    def train(self, env, num_episodes=1000):
        # Your AG3 training logic here
        pass

    def save(self, actor_path='actor.pth', critic_path='critic.pth'):
        try:
            actor_dir = os.path.dirname(actor_path)
            if actor_dir and not os.path.exists(actor_dir):
                os.makedirs(actor_dir)
            critic_dir = os.path.dirname(critic_path)
            if critic_dir and not os.path.exists(critic_dir):
                os.makedirs(critic_dir)
            torch.save(self.actor.state_dict(), actor_path)
            torch.save(self.critic.state_dict(), critic_path)
            print(f"Models saved to {actor_path} and {critic_path}")
        except Exception as e:
            print(f"Could not save models. Error: {e}")

    def load(self, actor_path='actor.pth', critic_path='critic.pth'):
        try:
            if not os.path.exists(actor_path) or not os.path.exists(critic_path):
                raise FileNotFoundError("Model files not found.")
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            print(f"Models loaded from {actor_path} and {critic_path}")
        except Exception as e:
            print(f"Could not load models. Error: {e}")


# Set up the environment
env = gym.make("BipedalWalker-v3", render_mode="human")
#env = gym.make("BipedalWalker-v3")

# Constants
num_episodes = 1000
PATH = 'data/'
PREFIX = 'bipedal_walker_ag3_v01'

# Initialize the PPOAgent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = A3CAgent(state_dim, action_dim)


# Load the agents
agent.load(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')

# Train the agent
agent.train(env, num_episodes)

# Save the agents
agent.save(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')


