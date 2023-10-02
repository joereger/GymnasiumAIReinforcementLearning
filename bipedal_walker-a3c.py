import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import pygame
import os
import numpy as np
import matplotlib.pyplot as plt

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

        # Initialization with random weights and biases drawn from a normal distribution
        init.normal_(self.fc1.weight, mean=0., std=1)
        init.normal_(self.fc1.bias, mean=0., std=1)
        init.normal_(self.fc2.weight, mean=0., std=1)
        init.normal_(self.fc2.bias, mean=0., std=1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor output
        actor_output = torch.tanh(self.actor(x))
        
        # Critic output
        critic_output = self.critic(x)
        
        return actor_output, critic_output


class A3CAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, which_optimizer='adam'):
        self.a3c_network = A3CNetwork(state_dim, action_dim)
        if which_optimizer == 'adam':
            self.optimizer = optim.Adam(self.a3c_network.parameters(), lr=learning_rate)
        elif which_optimizer == 'sgd':
            self.optimizer = optim.SGD(self.a3c_network.parameters(), lr=learning_rate)
        elif which_optimizer == 'adagrad':
            self.optimizer = optim.Adagrad(self.a3c_network.parameters(), lr=learning_rate)
        elif which_optimizer == 'adadelta':
            self.optimizer = optim.Adadelta(self.a3c_network.parameters(), lr=learning_rate)
        elif which_optimizer == 'nadam':
            self.optimizer = optim.NAdam(self.a3c_network.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.RMSprop(self.a3c_network.parameters(), lr=learning_rate)    

    def train(self, env, num_episodes=1000, gamma=0.99):
        best_episode_reward = float('-inf')  # Initialize with negative infinity
        total_rewards_across_episodes = 0  # Initialize total rewards across episodes

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0

            while not done and not truncated:
                action_mean, state_value = self.a3c_network(torch.FloatTensor(state))
                action_distribution = torch.distributions.Normal(action_mean, torch.tensor([0.1, 0.1, 0.1, 0.1]))
                actual_action = action_distribution.sample()
                next_state, reward, done, truncated, _ = env.step(actual_action.detach().numpy())
                next_state_value = self.a3c_network(torch.FloatTensor(next_state))[1].detach().numpy()
                advantage = reward + gamma * next_state_value * (1 - int(done)) - state_value.detach().numpy()
                target_value = torch.tensor([reward + gamma * next_state_value * (1 - int(done))]).view(state_value.shape)
                critic_loss = F.mse_loss(state_value, target_value)
                log_prob = action_distribution.log_prob(actual_action)
                actor_loss = -(log_prob * torch.tensor(advantage, dtype=torch.float32)).sum()
                total_loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.a3c_network.parameters(), max_norm=1)
                self.optimizer.step()
                state = next_state
                episode_reward += reward

            total_rewards_across_episodes += episode_reward  # Accumulate episode reward

            if episode_reward > best_episode_reward:
                best_episode_reward = episode_reward

            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {episode_reward}, Best Episode Reward: {best_episode_reward}")

        average_reward_across_episodes = total_rewards_across_episodes / num_episodes  # Calculate average reward
        print(f"Average Reward Across Episodes: {average_reward_across_episodes}")

        return best_episode_reward, average_reward_across_episodes  # Return both best and average rewards



    def load(self, model_path='_neural_network.pth'):
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found.")
            self.a3c_network.load_state_dict(torch.load(model_path))
            print(f"Modes loaded from {model_path}")
        except Exception as e:
            print(f"Could not load models. Error: {e}")
            
    def save(self, model_path='_neural_network.pth'):
        try:
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(self.a3c_network.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Could not save models. Error: {e}")

# Set up the environment
#env = gym.make("BipedalWalker-v3", render_mode="human")
env = gym.make("BipedalWalker-v3")

# Constants
num_episodes = 1000
PATH = 'data/'
PREFIX = 'bipedal_walker_a3c_v03'

# Initialize the A3C agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
a3c_model = A3CAgent(state_dim, action_dim)

# Load the model
#a3c_model.load(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')
# Train the model
#a3c_model.train(env, num_episodes)
# Save the model
#a3c_model.save(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')

class GridSearchResult:
    def __init__(self, learning_rate, gamma, which_optimizer, best_episode_reward, average_reward_across_episodes):
        self.learning_rate = learning_rate
        self.which_optimizer = which_optimizer
        self.gamma = gamma
        self.best_episode_reward = best_episode_reward
        self.average_reward_across_episodes = average_reward_across_episodes

# Initialize a list to store the results
grid_search_results = []

# Define the hyperparameters to test
learning_rate_values = [0.006] # [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4]
gamma_values = [0.6] # [0.8, 0.9, 0.95, 0.99, 0.999]
which_optimizer_values = ['rmsprop'] # ['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'nadam']

# Perform the grid search
for learning_rate in learning_rate_values:
    for gamma in gamma_values:
        for which_optimizer in which_optimizer_values:
            print(f"Testing learning_rate={learning_rate}, gamma={gamma}...")
            
            # Initialize your A3C agent here with the current hyperparameters
            agent = A3CAgent(state_dim, action_dim, learning_rate, which_optimizer)
            
            # Train the agent and get the best episode reward
            best_episode_reward, average_reward_across_episodes = agent.train(env, num_episodes=10000, gamma=gamma)
            
            # Store the result
            result = GridSearchResult(learning_rate, gamma, which_optimizer, best_episode_reward, average_reward_across_episodes)
            grid_search_results.append(result)
            
            print(f"Result: learning_rate={learning_rate}, gamma={gamma}, which_optimizer={which_optimizer}: best_episode: {best_episode_reward} avg_reward: {average_reward_across_episodes}")

# Create sexy plots (matplotlib for this)
plt.figure(figsize=(10, 6))
marker_dict = {'adam': 'o', 'sgd': 's', 'adagrad': '^', 'adadelta': 'D', 'nadam': 'x', 'ftrl': 'p', 'rmsprop': '+'}
line_dict = {}

for result in grid_search_results:
    print(f"Learning Rate: {result.learning_rate}, Gamma: {result.gamma}, Optimizer: {result.which_optimizer}, Best Episode Reward: {result.best_episode_reward}")
    
    marker = marker_dict.get(result.which_optimizer, 'o')
    line, = plt.plot(result.learning_rate, result.average_reward_across_episodes, marker=marker, label=result.which_optimizer if result.which_optimizer not in line_dict else "")
    
    if result.which_optimizer not in line_dict:
        line_dict[result.which_optimizer] = line

plt.xlabel('Learning Rate')
plt.ylabel('Avg Episode Reward')
plt.title('Grid Search Results')
plt.legend()
plt.show()



