import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from torch.distributions import Normal

# Actor Network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # Increased size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# Critic Network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # Increased size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPOAgent:
    def __init__(self, state_dim, action_dim, buffer_size=10000, learning_rate_actor=0.0003, learning_rate_critic=0.001, gamma=0.99, clip_epsilon=0.2):
        # Init
        self.actor_log_std = torch.tensor(0.0)  # Make sure to convert it to a tensor
        
        # Initialize Actor and Critic networks
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        
        # Initialize hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        # Initialize memory buffer (you can use a list or custom data structure)
        self.memory = []

        # Initialize memory buffer with a fixed size
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

    def train(self, env, num_episodes=1000, epochs=10):
        episode_rewards = []  # To store cumulative rewards for each episode

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False  # Initialize done flag

            while not done:  # Run until natural termination
                action = self.actor(torch.FloatTensor(state)).detach().numpy()
                next_state, reward, done, _, _ = env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

            # After collecting enough experiences, update the policy
            for _ in range(epochs):
                advantages, value_targets = self.compute_advantages()
                experiences = list(self.memory)
                states, actions, rewards, next_states, dones = zip(*experiences)
                self.update_policy(list(states), list(actions), advantages, value_targets)

            episode_rewards.append(episode_reward)
            print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward}")

        return episode_rewards


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
env = gym.make("BipedalWalker-v3", render_mode="human")
#env = gym.make("BipedalWalker-v3")

# Initialize the PPOAgent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPOAgent(state_dim, action_dim)

# Train the agent
agent.train(env)

# Evaluate the training
evaluate(agent, env)

