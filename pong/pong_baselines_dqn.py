"""
Pong Baselines-style DQN Implementation

This script implements a Double DQN with the standard environment wrappers and
techniques used in successful Atari RL implementations like OpenAI Baselines.

Key differences from our previous implementations:
1. Proper Atari wrappers (NoopReset, MaxAndSkip, EpisodicLife, FireReset)
2. Different observation processing order
3. Limited action space to only the relevant actions for Pong
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import time
import argparse
import ale_py
import json
import matplotlib.pyplot as plt
from collections import deque, Counter

# Import the environment wrappers from diagnostic script
from pong_diagnostic import (
    NoopResetEnv, MaxAndSkipEnv, FireResetEnv, EpisodicLifeEnv,
    WarpFrame, ClipRewardEnv, FrameStack, ScaledFloatFrame,
    make_atari_env
)

# --- Device Setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# --- DQN Network ---
class BaselinesAtariDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        """
        Standard DQN architecture used in Baselines.
        
        Args:
            input_shape: Shape of the input observations (C, H, W)
            n_actions: Number of possible actions
        """
        super(BaselinesAtariDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_output(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output(self, shape):
        """Calculate the output size of the conv layers."""
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """Forward pass through the network."""
        # x shape: (batch, channels, height, width)
        conv_out = self.conv(x)
        # Flatten the conv output
        conv_out = conv_out.view(conv_out.size(0), -1)
        return self.fc(conv_out)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """Sample a batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        
        # Convert to torch tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones

# --- Agent ---
class DoubleDQNAgent:
    def __init__(self, state_shape, n_actions, buffer_size=100000, 
                 learning_rate=1e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=10**5,
                 target_update=1000, batch_size=32):
        """
        Double DQN Agent with standard Baselines-style parameters
        
        Args:
            state_shape: Shape of the state (channels, height, width)
            n_actions: Number of possible actions
            buffer_size: Size of replay buffer
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Starting value of epsilon for exploration
            epsilon_final: Final value of epsilon
            epsilon_decay: Number of frames to decay epsilon over
            target_update: Number of frames between target network updates
            batch_size: Size of minibatch for training
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        
        # Create networks
        self.online_net = BaselinesAtariDQN(state_shape, n_actions).to(device)
        self.target_net = BaselinesAtariDQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target network doesn't need gradients
        
        # Create optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Initialize counters
        self.frame_count = 0
        self.update_count = 0
        
        # For statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.episode_q_values = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_losses = []
        self.current_episode_q_values = []
    
    def update_epsilon(self):
        """Update epsilon value based on frame count."""
        self.epsilon = max(
            self.epsilon_final, 
            self.epsilon - (self.epsilon - self.epsilon_final) / self.epsilon_decay
        )
    
    def act(self, state, deterministic=False):
        """Select an action using epsilon-greedy policy."""
        if not deterministic and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)
        
        # Get Q-values from online network
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        
        # Keep track of average Q-value for statistics
        self.current_episode_q_values.append(q_values.max().item())
        
        # Select action with highest Q-value
        return q_values.argmax(dim=1).item()
    
    def learn(self, state, action, reward, next_state, done):
        """Store experience and perform learning if buffer is large enough."""
        # Store experience in replay buffer
        self.buffer.append((state, action, reward, next_state, done))
        
        # Update counters
        self.frame_count += 1
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Update epsilon
        self.update_epsilon()
        
        # Only learn if buffer has enough samples
        if len(self.buffer) >= self.batch_size:
            # Sample batch from replay buffer
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            
            # Get current Q values
            current_q_values = self.online_net(states).gather(1, actions)
            
            # Double DQN: use online network to select actions, target network to evaluate
            with torch.no_grad():
                # Select actions using online network
                next_actions = self.online_net(next_states).max(1)[1].unsqueeze(1)
                # Evaluate actions using target network
                next_q_values = self.target_net(next_states).gather(1, next_actions)
                # Compute target Q values
                target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            
            # Keep track of loss for statistics
            self.current_episode_losses.append(loss.item())
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to stabilize training
            for param in self.online_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            
            # Update target network if needed
            if self.frame_count % self.target_update == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())
                self.update_count += 1
                print(f"Target network updated ({self.update_count})")
        
        return self.frame_count
    
    def end_episode(self):
        """Record statistics at the end of an episode."""
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        if self.current_episode_losses:
            self.episode_losses.append(np.mean(self.current_episode_losses))
        if self.current_episode_q_values:
            self.episode_q_values.append(np.mean(self.current_episode_q_values))
        
        # Reset episode statistics
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_episode_losses = []
        self.current_episode_q_values = []
    
    def get_state_dict(self):
        """Get state dict for saving."""
        return {
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'frame_count': self.frame_count,
            'update_count': self.update_count,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_losses': self.episode_losses,
            'episode_q_values': self.episode_q_values
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for loading."""
        self.online_net.load_state_dict(state_dict['online_net'])
        self.target_net.load_state_dict(state_dict['target_net'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.frame_count = state_dict['frame_count']
        self.update_count = state_dict['update_count']
        self.epsilon = state_dict['epsilon']
        self.episode_rewards = state_dict['episode_rewards']
        self.episode_lengths = state_dict['episode_lengths']
        if 'episode_losses' in state_dict:
            self.episode_losses = state_dict['episode_losses']
        if 'episode_q_values' in state_dict:
            self.episode_q_values = state_dict['episode_q_values']

# --- Training Function ---
def train(env, agent, num_frames=1_000_000, eval_freq=10000, save_freq=10000, render=False):
    """Train the agent."""
    frame_idx = 0
    eval_rewards = []
    eval_frame_indices = []
    
    # For plotting
    history = {
        "frames": [],
        "rewards": [],
        "lengths": [],
        "epsilons": [],
        "losses": [],
        "q_values": [],
        "eval_frames": [],
        "eval_rewards": []
    }
    
    # Make sure target folder exists
    os.makedirs("data/pong/baselines", exist_ok=True)
    
    print("Starting training...")
    episode = 0
    
    while frame_idx < num_frames:
        episode += 1
        state, info = env.reset()
        done = False
        
        while not done:
            # Render if needed
            if render:
                env.render()
            
            # Select and perform action
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Learn from experience
            frame_idx = agent.learn(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            
            # Evaluate and save if needed
            if frame_idx % eval_freq == 0:
                eval_reward = evaluate(agent, make_atari_env("PongNoFrameskip-v4"), num_episodes=5, render=False)
                eval_rewards.append(eval_reward)
                eval_frame_indices.append(frame_idx)
                print(f"Frame {frame_idx}: Evaluation reward = {eval_reward:.2f}")
                
                # Save history for plotting
                history["frames"].append(frame_idx)
                history["rewards"].append(np.mean(agent.episode_rewards[-100:]) if agent.episode_rewards else 0)
                history["lengths"].append(np.mean(agent.episode_lengths[-100:]) if agent.episode_lengths else 0)
                history["epsilons"].append(agent.epsilon)
                history["losses"].append(np.mean(agent.episode_losses[-100:]) if agent.episode_losses else 0)
                history["q_values"].append(np.mean(agent.episode_q_values[-100:]) if agent.episode_q_values else 0)
                history["eval_frames"].append(frame_idx)
                history["eval_rewards"].append(eval_reward)
                
                # Plot results
                plot_results(history, f"data/pong/baselines/dqn_results_{frame_idx}.png")
                
                # Save model
                torch.save(agent.get_state_dict(), f"data/pong/baselines/dqn_model_{frame_idx}.pth")
                
                # Save history
                with open(f"data/pong/baselines/dqn_history_{frame_idx}.json", "w") as f:
                    json.dump(history, f)
            
            # Save if needed
            if frame_idx % save_freq == 0:
                torch.save(agent.get_state_dict(), f"data/pong/baselines/dqn_model_latest.pth")
        
        # End of episode
        agent.end_episode()
        
        # Print episode stats
        if episode % 1 == 0:
            print(f"Episode {episode}: Reward = {agent.episode_rewards[-1]:.2f}, Length = {agent.episode_lengths[-1]}, Epsilon = {agent.epsilon:.4f}")
    
    print("Training completed.")
    return history

# --- Evaluation Function ---
def evaluate(agent, env, num_episodes=5, render=False):
    """Evaluate the agent."""
    total_reward = 0
    
    for i in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Render if needed
            if render:
                env.render()
            
            # Select action
            action = agent.act(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        total_reward += episode_reward
        print(f"Evaluation episode {i+1}: Reward = {episode_reward:.2f}")
    
    avg_reward = total_reward / num_episodes
    return avg_reward

# --- Plotting Function ---
def plot_results(history, filename):
    """Plot the training results."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    axs[0, 0].plot(history["frames"], history["rewards"], label="Training (100-ep avg)")
    if history["eval_frames"]:
        axs[0, 0].plot(history["eval_frames"], history["eval_rewards"], label="Evaluation", linestyle="--")
    axs[0, 0].set_title("Rewards")
    axs[0, 0].set_xlabel("Frames")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot epsilon
    axs[0, 1].plot(history["frames"], history["epsilons"])
    axs[0, 1].set_title("Epsilon")
    axs[0, 1].set_xlabel("Frames")
    axs[0, 1].set_ylabel("Epsilon")
    axs[0, 1].grid(True)
    
    # Plot losses
    if history["losses"]:
        axs[1, 0].plot(history["frames"], history["losses"])
        axs[1, 0].set_title("Loss")
        axs[1, 0].set_xlabel("Frames")
        axs[1, 0].set_ylabel("Loss")
        axs[1, 0].grid(True)
    
    # Plot Q-values
    if history["q_values"]:
        axs[1, 1].plot(history["frames"], history["q_values"])
        axs[1, 1].set_title("Average Q-Value")
        axs[1, 1].set_xlabel("Frames")
        axs[1, 1].set_ylabel("Q-Value")
        axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train a DQN agent on Pong using standard Baselines wrappers")
    parser.add_argument("--frames", type=int, default=1_000_000, help="Number of frames to train for")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--eval_freq", type=int, default=10000, help="Frequency of evaluation")
    parser.add_argument("--save_freq", type=int, default=10000, help="Frequency of saving")
    parser.add_argument("--load", action="store_true", help="Load the model")
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("data/pong/baselines", exist_ok=True)
    
    # Create environment
    env = make_atari_env("PongNoFrameskip-v4", render_mode="human" if args.render else None)
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Create agent
    agent = DoubleDQNAgent(
        state_shape=(4, 84, 84),  # CNN expects channels first
        n_actions=env.action_space.n,
        buffer_size=100000,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=100000,
        target_update=1000,
        batch_size=32
    )
    
    # Load model if needed
    if args.load:
        if os.path.exists("data/pong/baselines/dqn_model_latest.pth"):
            print("Loading model...")
            agent.load_state_dict(torch.load("data/pong/baselines/dqn_model_latest.pth"))
            print(f"Loaded model with {agent.frame_count} frames")
        else:
            print("No model to load.")
    
    # Train agent
    history = train(
        env=env,
        agent=agent,
        num_frames=args.frames,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        render=args.render
    )
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
