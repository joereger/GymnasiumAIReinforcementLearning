import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.distributions import Normal
import pygame
import os
from visualization_utils import * # mah little helper utils for visualizing activations

# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)  
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)

        init.normal_(self.fc1.weight, mean=0., std=1)
        init.normal_(self.fc1.bias, mean=0., std=1)
        init.normal_(self.fc2.weight, mean=0., std=1)
        init.normal_(self.fc2.bias, mean=0., std=1)
        init.normal_(self.fc3.weight, mean=0., std=1)
        init.normal_(self.fc3.bias, mean=0., std=1)
        init.normal_(self.fc4.weight, mean=0., std=1)
        init.normal_(self.fc4.bias, mean=0., std=1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)  
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        init.normal_(self.fc1.weight, mean=0., std=1)
        init.normal_(self.fc1.bias, mean=0., std=1)
        init.normal_(self.fc2.weight, mean=0., std=1)
        init.normal_(self.fc2.bias, mean=0., std=1)
        init.normal_(self.fc3.weight, mean=0., std=1)
        init.normal_(self.fc3.bias, mean=0., std=1)
        init.normal_(self.fc4.weight, mean=0., std=1)
        init.normal_(self.fc4.bias, mean=0., std=1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class PPOAgent:
    #def __init__(self, state_dim, action_dim, buffer_size=100000, learning_rate_actor=0.0003, learning_rate_critic=0.001, gamma=0.99, clip_epsilon=0.2):
    def __init__(self, state_dim, action_dim, buffer_size=10000, learning_rate_actor=0.0001, learning_rate_critic=0.01, gamma=0.99, clip_epsilon=0.2):
        # Init
        self.actor_log_std = torch.tensor(0.0)  # Make sure to convert it to a tensor
        
        # Initialize Actor and Critic networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        # Initialize optimizers
        #self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        #self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=learning_rate_critic)
        
        # Initialize hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        # Initialize memory buffer (you can use a list or custom data structure)
        self.memory = []

        # Initialize memory buffer with a fixed size
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)

    def store_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        #print('print(len(self.memory)): ', len(self.memory))

    def compute_advantages(self, gamma=0.99, tau=0.95):
        advantages = []
        value_targets = []
        next_value = 0  # Initialize next_value as 0 for the terminal state
        next_advantage = 0  # Initialize next_advantage as 0

        # Loop through memory in reverse order
        for state, action, reward, next_state, done in reversed(self.memory):
            # Get value estimate for the current state from the Critic network
            value = self.critic(torch.FloatTensor(state))

            # Calculate the TD error
            td_error = reward + gamma * next_value * (1 - done) - value

            # Calculate the advantage using GAE
            advantage = td_error + gamma * tau * next_advantage * (1 - done)

            # Update the next_value and next_advantage
            next_value = value
            next_advantage = advantage

            # Store the calculated advantage and value target
            advantages.append(advantage)
            value_targets.append(reward + gamma * next_value * (1 - done))

        # Reverse the lists and convert to PyTorch tensors
        advantages = torch.FloatTensor(advantages[::-1])
        value_targets = torch.FloatTensor(value_targets[::-1])

        return advantages, value_targets

    def update_policy(self, states, actions, advantages, value_targets):
        # Convert to tensors if they aren't already
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        advantages = torch.FloatTensor(advantages)
        value_targets = torch.FloatTensor(value_targets)
        
        # Update Critic
        value_preds = self.critic(states)
        critic_loss = F.mse_loss(value_preds.squeeze(1), value_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        mean = self.actor(states)  # Assuming your actor network outputs the mean
        std = torch.exp(self.actor_log_std)  # log_std could be a learnable parameter or fixed
        
        # Create a normal distribution object
        normal = Normal(mean, std)
        
        # Compute log-probability of the taken actions
        log_prob = normal.log_prob(actions)
        
        # Expand dimensions of advantages to make it [64, 1]
        expanded_advantages = advantages.unsqueeze(-1)

        # Calculate actor loss
        actor_loss = -(log_prob * expanded_advantages).mean()

        # Perform the optimization step
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #print(f"Critic Loss: {critic_loss.item():.4f}, Actor Loss: {actor_loss.item():.4f}, Mean Action: {mean.mean().item():.4f}")


    def train(self, env, num_episodes=1000, epochs=10):
        episode_rewards = []  # To store cumulative rewards for each episode

        for episode in range(num_episodes):
            state, _ = env.reset()
            self.memory = deque(maxlen=self.buffer_size) # Clear the memory buffer each episode
            episode_reward = 0
            done = False  
            truncated = False
            timesteps = 0  

            while not done and not truncated:  # Run until natural termination
                action = self.actor(torch.FloatTensor(state)).detach().numpy()
                #print('timesteps: ', timesteps, 'state: ', state, ' torch.FloatTensor(state): ', torch.FloatTensor(state))
                next_state, reward, done, truncated, info = env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                timesteps += 1
                #print(f"ep: {episode}  t: {timesteps}  reward: {reward:.4f}  ep_reward: {episode_reward:.4f}  action: {action}")

            # After collecting enough experiences, update the policy
            for _ in range(epochs):
                advantages, value_targets = self.compute_advantages()
                experiences = list(self.memory)
                states, actions, rewards, next_states, dones = zip(*experiences)
                self.update_policy(list(states), list(actions), advantages, value_targets)

            episode_rewards.append(episode_reward)
            print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward}")

            # Visualize the activations
            plot_activations(activations)

        return episode_rewards
    
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

# Visualization
plt.ion()
plt.axis('off')
plt.show()
activations = {}
register_hooks(agent.actor, activations)

# Load the agents
agent.load(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')

# Train the agent
agent.train(env, num_episodes)

# Save the agents
agent.save(PATH + PREFIX + '_actor.pth', PATH + PREFIX + '_critic.pth')

# Evaluate the training
#evaluate(agent, env)

