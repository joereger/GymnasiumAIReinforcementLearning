"""
Pong Fixed PPO Implementation

This script implements a PPO agent for the Pong environment with proper environment wrappers
and preprocessing based on standard practices from successful implementations.

Key fixes from previous version:
1. Proper Atari wrappers (NoopReset, MaxAndSkip, EpisodicLife, FireReset)
2. Correct input shape and channels-first ordering for PyTorch
3. Fixed frame stacking and preprocessing
4. Standard hyperparameters from successful implementations
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import time
import argparse
import json
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from torch.distributions import Categorical

# Import environment wrappers from diagnostic script
from pong_diagnostic import (
    NoopResetEnv, MaxAndSkipEnv, FireResetEnv, EpisodicLifeEnv,
    WarpFrame, ClipRewardEnv, FrameStack, ScaledFloatFrame, 
    make_atari_env
)

# --- Device setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# --- Actor-Critic Network Architecture ---
class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    Properly implements channels-first approach for PyTorch.
    """
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        
        # Calculate input channels from input shape (channels first)
        self.channels = input_shape[0]
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate flattened size
        conv_out_size = self._get_conv_output(input_shape)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_output(self, shape):
        """Calculate output size of conv features."""
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """Forward pass through network."""
        # x input shape: (batch, channels, height, width)
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Get policy logits and value
        policy_logits = self.actor(features)
        value = self.critic(features)
        
        # Apply softmax to get action probabilities
        policy = F.softmax(policy_logits, dim=-1)
        
        return policy, value
    
    def evaluate(self, states, actions):
        """
        Evaluate actions and compute:
        - Log probabilities of taken actions
        - Value function estimates
        - Action distribution entropy
        """
        policy, values = self(states)
        
        # Create a distribution from policy
        dist = Categorical(policy)
        
        # Get log probs of actions
        action_log_probs = dist.log_prob(actions)
        
        # Calculate entropy of the action distribution
        entropy = dist.entropy().mean()
        
        return action_log_probs, values.squeeze(-1), entropy

# --- PPO Agent ---
class PPOAgent:
    def __init__(self, 
                 input_shape, 
                 n_actions, 
                 lr=2.5e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_param=0.1,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 ppo_epochs=4,
                 batch_size=64,
                 buffer_size=2048):
        """
        PPO Agent implementation.
        
        Args:
            input_shape: Shape of state input (channels, height, width)
            n_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max norm for gradient clipping
            ppo_epochs: Number of PPO epochs per update
            batch_size: Minibatch size for updates
            buffer_size: Size of rollout buffer before update
        """
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Create actor-critic network
        self.network = ActorCritic(input_shape, n_actions).to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        # Initialize rollout buffer storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # For tracking stats
        self.episode_rewards = []
        self.training_info = {
            "policy_losses": [],
            "value_losses": [],
            "entropy_losses": [],
            "total_losses": []
        }
    
    def act(self, state, deterministic=False):
        """
        Select an action given the current state.
        
        Args:
            state: Current state observation
            deterministic: If True, select best action for evaluation
            
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
            value: Value estimate for current state
        """
        # Ensure state has correct shape (channels first for PyTorch)
        if len(state.shape) == 3 and state.shape[0] != self.input_shape[0]:
            # Convert from (H, W, C) to (C, H, W) if needed
            state = np.transpose(state, (2, 0, 1))
        
        # Add batch dimension and convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Forward pass through network
        with torch.no_grad():
            policy, value = self.network(state_tensor)
        
        # Select action
        if deterministic:
            # For evaluation, select most probable action
            action = policy.argmax(dim=1).item()
            log_prob = torch.log(policy[0, action]).item()
        else:
            # Sample from action distribution
            dist = Categorical(policy)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor([action], device=device)).item()
        
        return action, log_prob, value.item()
    
    def store_transition(self, state, action, reward, done, log_prob, value):
        """Store a transition in the rollout buffer."""
        # Ensure state has correct shape
        if len(state.shape) == 3 and state.shape[0] != self.input_shape[0]:
            # Convert from (H, W, C) to (C, H, W) if needed
            state = np.transpose(state, (2, 0, 1))
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def normalize(self, x):
        """Normalize a vector."""
        x = np.array(x)
        if len(x) > 1:
            return (x - x.mean()) / (x.std() + 1e-8)
        return x
    
    def compute_gae(self, last_value):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            last_value: Value estimate for the state after the last state in buffer
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns (advantages + values)
        """
        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones)
        
        # Initialize advantages and returns
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Compute GAE
        gae = 0
        for t in reversed(range(len(rewards))):
            # Set delta based on reward, discount, next value, and done flag
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            # Update GAE with discounted delta values
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Compute returns (advantages + values)
        returns = advantages + np.array(self.values)
        
        return advantages, returns
    
    def update(self, last_value):
        """
        Update policy using the collected rollout data.
        
        Args:
            last_value: Value estimate for the state after the last state in buffer
            
        Returns:
            info_dict: Dictionary of loss statistics
        """
        # Compute advantages and returns
        advantages, returns = self.compute_gae(last_value)
        
        # Normalize advantages (helps stabilize training)
        advantages = self.normalize(advantages)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(device)
        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)
        
        # PPO update loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # Multiple epochs of updates
        for _ in range(self.ppo_epochs):
            # Generate random indices for minibatches
            indices = np.random.permutation(len(self.states))
            
            # Process minibatches
            for start_idx in range(0, len(indices), self.batch_size):
                # Get minibatch indices
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                
                # Extract batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get new log probs, values, and entropy
                new_log_probs, values, entropy = self.network.evaluate(batch_states, batch_actions)
                
                # Compute policy loss (PPO clip objective)
                # Ratio between new and old policies
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss with clipping
                value_pred_clipped = batch_returns + torch.clamp(
                    values - batch_returns, -self.clip_param, self.clip_param
                )
                value_loss1 = F.mse_loss(values, batch_returns)
                value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                # Clip gradients
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Compute average losses
        num_updates = len(indices) // self.batch_size + int(len(indices) % self.batch_size > 0)
        avg_policy_loss = total_policy_loss / (self.ppo_epochs * num_updates)
        avg_value_loss = total_value_loss / (self.ppo_epochs * num_updates)
        avg_entropy_loss = total_entropy_loss / (self.ppo_epochs * num_updates)
        avg_total_loss = avg_policy_loss + self.value_coef * avg_value_loss + self.entropy_coef * avg_entropy_loss
        
        # Store training info
        self.training_info["policy_losses"].append(avg_policy_loss)
        self.training_info["value_losses"].append(avg_value_loss)
        self.training_info["entropy_losses"].append(-avg_entropy_loss.item())  # Flip sign for clarity
        self.training_info["total_losses"].append(avg_total_loss)
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
        # Return training info
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": -avg_entropy_loss,
            "total_loss": avg_total_loss
        }
    
    def save(self, path):
        """Save the model and training info."""
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_info": self.training_info,
            "episode_rewards": self.episode_rewards
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the model and training info."""
        checkpoint = torch.load(path, map_location=device)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "training_info" in checkpoint:
            self.training_info = checkpoint["training_info"]
        if "episode_rewards" in checkpoint:
            self.episode_rewards = checkpoint["episode_rewards"]
        
        print(f"Model loaded from {path}")

# --- Training Function ---
def train(agent, env, n_steps=1_000_000, eval_freq=10_000, save_freq=10_000, render=False):
    """
    Train the PPO agent.
    
    Args:
        agent: PPO agent
        env: Environment
        n_steps: Total number of training steps
        eval_freq: Frequency of evaluation
        save_freq: Frequency of saving
        render: Whether to render the environment during training
    """
    # Create directories
    os.makedirs("data/pong/fixed_ppo", exist_ok=True)
    
    # Track statistics
    steps_done = 0
    episode_count = 0
    best_eval_reward = -float('inf')
    
    # Time tracking
    start_time = time.time()
    
    # Stats dictionary for plotting
    stats = {
        "steps": [],
        "rewards": [],
        "avg_rewards": [],
        "policy_losses": [],
        "value_losses": [],
        "entropies": [],
        "eval_steps": [],
        "eval_rewards": []
    }
    
    print("Starting training...")
    
    while steps_done < n_steps:
        episode_count += 1
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        # For tracking rollout buffer
        rollout_length = 0
        
        while not done:
            # Select action
            action, log_prob, value = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, done, log_prob, value)
            
            # Update counters
            episode_reward += reward
            steps_done += 1
            rollout_length += 1
            
            # Move to next state
            state = next_state
            
            # Render if needed
            if render:
                env.render()
            
            # Check if buffer is full or episode ended
            if rollout_length >= agent.buffer_size or done:
                # Calculate last value if not done
                if not done:
                    _, _, last_value = agent.act(state)
                else:
                    last_value = 0.0
                
                # Update policy
                loss_info = agent.update(last_value)
                rollout_length = 0
                
                print(f"Step {steps_done}/{n_steps} | "
                      f"Policy Loss: {loss_info['policy_loss']:.4f} | "
                      f"Value Loss: {loss_info['value_loss']:.4f} | "
                      f"Entropy: {loss_info['entropy']:.4f}")
            
            # Evaluate periodically
            if steps_done % eval_freq == 0:
                eval_reward = evaluate(agent, make_atari_env("PongNoFrameskip-v4"), num_episodes=5)
                
                # Save stats
                stats["eval_steps"].append(steps_done)
                stats["eval_rewards"].append(eval_reward)
                
                # Check if best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    agent.save(f"data/pong/fixed_ppo/ppo_best_model.pth")
                    print(f"New best evaluation reward: {best_eval_reward}")
                    
                # Plot progress
                plot_progress(stats, f"data/pong/fixed_ppo/ppo_progress.png")
            
            # Save periodically
            if steps_done % save_freq == 0:
                agent.save(f"data/pong/fixed_ppo/ppo_step_{steps_done}.pth")
                
                # Save stats
                with open(f"data/pong/fixed_ppo/ppo_stats.json", "w") as f:
                    json.dump(stats, f)
            
            # Check if training completed
            if steps_done >= n_steps:
                break
        
        # Episode completed
        agent.episode_rewards.append(episode_reward)
        
        # Update stats
        stats["steps"].append(steps_done)
        stats["rewards"].append(episode_reward)
        stats["avg_rewards"].append(np.mean(agent.episode_rewards[-100:]))
        if agent.training_info["policy_losses"]:
            stats["policy_losses"].append(agent.training_info["policy_losses"][-1])
        if agent.training_info["value_losses"]:
            stats["value_losses"].append(agent.training_info["value_losses"][-1])
        if agent.training_info["entropy_losses"]:
            stats["entropies"].append(agent.training_info["entropy_losses"][-1])
        
        # Print episode stats
        elapsed_time = time.time() - start_time
        print(f"Episode {episode_count} | "
              f"Steps: {steps_done}/{n_steps} | "
              f"Reward: {episode_reward:.2f} | "
              f"Avg100: {np.mean(agent.episode_rewards[-100:]):.2f} | "
              f"Elapsed: {elapsed_time:.2f}s")
    
    # Save final model
    agent.save(f"data/pong/fixed_ppo/ppo_final_model.pth")
    
    # Save final stats
    with open(f"data/pong/fixed_ppo/ppo_stats_final.json", "w") as f:
        json.dump(stats, f)
    
    # Final plot
    plot_progress(stats, f"data/pong/fixed_ppo/ppo_progress_final.png")
    
    print(f"Training completed. Best evaluation reward: {best_eval_reward}")
    return stats

# --- Evaluation Function ---
def evaluate(agent, env, num_episodes=5, render=False):
    """
    Evaluate the PPO agent.
    
    Args:
        agent: PPO agent
        env: Environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment during evaluation
        
    Returns:
        mean_reward: Mean reward over episodes
    """
    total_reward = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action deterministically
            action, _, _ = agent.act(state, deterministic=True)
            
            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Render if needed
            if render:
                env.render()
        
        total_reward += episode_reward
        print(f"Evaluation episode {episode+1}/{num_episodes}: Reward = {episode_reward}")
    
    mean_reward = total_reward / num_episodes
    print(f"Mean evaluation reward: {mean_reward}")
    return mean_reward

# --- Plotting Function ---
def plot_progress(stats, filename):
    """
    Plot training progress.
    
    Args:
        stats: Dictionary of training statistics
        filename: Output filename
    """
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards and average
    plt.subplot(2, 2, 1)
    if stats["rewards"]:
        plt.plot(stats["steps"], stats["rewards"], alpha=0.3, label="Episode Reward")
    if stats["avg_rewards"]:
        plt.plot(stats["steps"], stats["avg_rewards"], label="100-episode Average")
    if stats["eval_rewards"]:
        plt.plot(stats["eval_steps"], stats["eval_rewards"], 'r--', label="Evaluation Reward")
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Training Rewards")
    plt.grid(True, alpha=0.3)
    
    # Plot policy loss
    plt.subplot(2, 2, 2)
    if stats["policy_losses"]:
        plt.plot(stats["steps"][:len(stats["policy_losses"])], stats["policy_losses"])
    plt.xlabel("Steps")
    plt.ylabel("Policy Loss")
    plt.title("Policy Loss")
    plt.grid(True, alpha=0.3)
    
    # Plot value loss
    plt.subplot(2, 2, 3)
    if stats["value_losses"]:
        plt.plot(stats["steps"][:len(stats["value_losses"])], stats["value_losses"])
    plt.xlabel("Steps")
    plt.ylabel("Value Loss")
    plt.title("Value Loss")
    plt.grid(True, alpha=0.3)
    
    # Plot entropy
    plt.subplot(2, 2, 4)
    if stats["entropies"]:
        plt.plot(stats["steps"][:len(stats["entropies"])], stats["entropies"])
    plt.xlabel("Steps")
    plt.ylabel("Entropy")
    plt.title("Entropy")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on Pong with proper wrappers")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Number of steps to train for")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    parser.add_argument("--eval", action="store_true", help="Evaluate the agent instead of training")
    parser.add_argument("--load", type=str, default=None, help="Path to load model from")
    args = parser.parse_args()
    
    # Create properly wrapped environment
    env = make_atari_env("PongNoFrameskip-v4", render_mode="human" if args.render else None)
    
    # Print environment info
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")
    
    # Check observation shape and adjust if necessary
    # Proper shape for PyTorch: (channels, height, width)
    input_shape = (4, 84, 84)  # 4 stacked frames, 84x84 grayscale images
    
    # Create agent
    agent = PPOAgent(
        input_shape=input_shape,
        n_actions=env.action_space.n,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_param=0.1,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        batch_size=64,
        buffer_size=2048
    )
    
    # Load model if specified
    if args.load:
        agent.load(args.load)
    elif args.eval and not args.load:
        # Try to load best model for evaluation
        if os.path.exists("data/pong/fixed_ppo/ppo_best_model.pth"):
            agent.load("data/pong/fixed_ppo/ppo_best_model.pth")
        else:
            print("No model found for evaluation. Training a new one.")
            args.eval = False
    
    # Evaluate or train
    if args.eval:
        print("Evaluating agent...")
        evaluate(agent, env, num_episodes=10, render=args.render)
    else:
        print("Training agent...")
        train(agent, env, n_steps=args.steps, render=args.render)
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()
