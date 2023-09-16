import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import pygame
import os
import numpy as np

class A3CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A3CNetwork, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)

        # Actor layers
        self.actor = nn.Linear(256, action_dim)
        
        # Critic layers
        self.critic = nn.Linear(256, 1)

        # Initialization code remains the same

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor output
        actor_output = torch.tanh(self.actor(x))
        
        # Critic output
        critic_output = self.critic(x)
        
        return actor_output, critic_output


class A3CAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.0001):
        self.a3c_network = A3CNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.a3c_network.parameters(), lr=learning_rate)

    def train(self, env, num_episodes=1000, gamma=0.99):
        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Forward pass to get the action probabilities and state value
                action_prob, state_value = self.a3c_network(torch.FloatTensor(state))
                action_prob = action_prob.detach().numpy()
                
                # Sample action from the action probabilities
                action = np.random.choice(len(action_prob), p=action_prob)
                
                # Take the action in the environment
                next_state, reward, done, _ = env.step(action)
                
                # Calculate the advantage
                next_state_value = self.a3c_network(torch.FloatTensor(next_state))[1].detach().numpy()
                advantage = reward + gamma * next_state_value * (1 - int(done)) - state_value.detach().numpy()
                
                # Calculate the loss
                critic_loss = F.mse_loss(state_value, torch.tensor([reward + gamma * next_state_value * (1 - int(done))]))
                actor_loss = -torch.log(action_prob[action]) * advantage
                total_loss = actor_loss + critic_loss
                
                # Perform a gradient step
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Update the state and episode reward
                state = next_state
                episode_reward += reward
            
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {episode_reward}")


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
PREFIX = 'bipedal_walker_a3c_v01'

# Initialize the A3C agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = A3CAgent(state_dim, action_dim)


# Load the agents
agent.load(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')

# Train the agent
agent.train(env, num_episodes)

# Save the agents
agent.save(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')


