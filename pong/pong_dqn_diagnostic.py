"""
Pong DQN Diagnostic Script

This file runs diagnostics on our DQN implementation to identify why it's not learning.
It focuses on tensor shapes, reward distribution, and action effectiveness.
"""

import gymnasium as gym
import torch
import numpy as np
import random
import os
import sys
import time
import matplotlib.pyplot as plt
from collections import Counter
import ale_py

# Import our existing code
from pong_dqn_utils import make_atari_env
from pong_dqn_model import DQNAgent, device

def print_shape_info(obj, name="Object"):
    """Print shape information about numpy arrays or torch tensors."""
    if isinstance(obj, np.ndarray):
        print(f"{name} (numpy): shape={obj.shape}, dtype={obj.dtype}, min={obj.min():.4f}, max={obj.max():.4f}")
    elif isinstance(obj, torch.Tensor):
        print(f"{name} (torch): shape={obj.shape}, dtype={obj.dtype}, min={obj.min().item():.4f}, max={obj.max().item():.4f}")
    else:
        print(f"{name}: type={type(obj)}")

def visualize_state(state, title="State"):
    """Visualize the state for analysis."""
    plt.figure(figsize=(15, 5))
    
    # If state is channels-first, reshape it to channels-last for visualization
    if len(state.shape) == 3 and state.shape[0] == 4:  # (4, 84, 84) format
        state_viz = np.transpose(state, (1, 2, 0))  # Convert to (84, 84, 4)
    else:
        state_viz = state
    
    # Plot each channel
    for i in range(state_viz.shape[-1]):
        plt.subplot(1, 4, i+1)
        plt.imshow(state_viz[..., i], cmap='gray')
        plt.title(f"Channel {i+1}")
        plt.axis('off')
    
    plt.suptitle(title)
    
    # Save the visualization
    os.makedirs('data/pong/diagnostics', exist_ok=True)
    plt.savefig(f'data/pong/diagnostics/state_viz_{int(time.time())}.png')
    plt.close()

# --- Action Space Reduction Wrapper ---
class ReducedActionSpaceWrapper(gym.ActionWrapper):
    """
    Reduces the action space to only 3 actions: NOOP, UP, DOWN.
    This simplifies learning for Pong dramatically.
    """
    def __init__(self, env):
        super().__init__(env)
        # Original actions: NOOP(0), FIRE(1), RIGHT(2), LEFT(3), RIGHTFIRE(4), LEFTFIRE(5)
        # Reduced: NOOP(0), UP(2), DOWN(3)
        self.valid_actions = [0, 2, 3]
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))
        print(f"Reduced action space from {env.action_space.n} to {self.action_space.n}")
        
        # Get action meanings for debug
        self.action_meanings = env.unwrapped.get_action_meanings()
        self.reduced_action_meanings = [self.action_meanings[a] for a in self.valid_actions]
        print(f"Original actions: {self.action_meanings}")
        print(f"Reduced actions: {self.reduced_action_meanings}")
    
    def action(self, action):
        """Map the reduced action index to the original action."""
        return self.valid_actions[action]

def make_diagnostic_env(env_id, reduced_actions=False, render_mode=None):
    """Create a properly wrapped environment with diagnostic info."""
    env = make_atari_env(env_id, render_mode=render_mode)
    
    if reduced_actions:
        env = ReducedActionSpaceWrapper(env)
    
    return env

def run_diagnostic_episode(env, agent, explore=True, num_steps=5000, visualize_every=500, print_every=100):
    """Run a diagnostic episode with detailed logging."""
    state, info = env.reset(seed=42)
    print_shape_info(state, "Initial State")
    visualize_state(state, "Initial State")
    
    # Initialize stats
    total_reward = 0
    rewards = []
    actions = []
    q_values_log = []
    step = 0
    done = False
    
    while not done and step < num_steps:
        # Get state representation for PyTorch
        if len(state.shape) == 3 and state.shape[0] != 4:  # If not already channels-first
            print(f"Step {step}: State needs transposing from {state.shape}")
            state_for_model = np.transpose(state, (2, 0, 1))
        else:
            state_for_model = state
        
        # Get Q-values for logging
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_for_model).unsqueeze(0).to(device)
            q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
            q_values_log.append(q_values)
        
        # Select action
        action = agent.act(state, explore=explore)
        actions.append(action)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        total_reward += reward
        done = terminated or truncated
        
        # Visualize periodically
        if step % visualize_every == 0:
            visualize_state(next_state, f"Step {step}, Action {action}, Reward {reward}")
        
        # Print diagnostics periodically
        if step % print_every == 0 or reward != 0:
            print(f"Step {step}: Action={action}, Reward={reward:.1f}, Q-values={q_values}")
        
        # Update state
        state = next_state
        step += 1
    
    # Calculate stats
    print("\nEpisode Stats:")
    print(f"Steps: {step}")
    print(f"Total reward: {total_reward}")
    print(f"Average Q-value magnitude: {np.mean(np.abs(q_values_log)):.6f}")
    print(f"Action distribution: {dict(Counter(actions))}")
    
    # Visualize action and reward distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(dict(Counter(actions)).keys(), dict(Counter(actions)).values())
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    
    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=3)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(f'data/pong/diagnostics/action_reward_dist_{int(time.time())}.png')
    plt.close()
    
    # Visualize Q-value evolution
    plt.figure(figsize=(10, 6))
    q_values_log = np.array(q_values_log)
    for a in range(q_values_log.shape[1]):
        plt.plot(q_values_log[:, a], label=f"Action {a}")
    plt.title("Q-value Evolution")
    plt.xlabel("Step")
    plt.ylabel("Q-value")
    plt.legend()
    plt.savefig(f'data/pong/diagnostics/q_values_{int(time.time())}.png')
    plt.close()
    
    return total_reward, actions, rewards, q_values_log

def main():
    print("=" * 80)
    print("PONG DQN DIAGNOSTIC")
    print("=" * 80)
    print("\nThis script diagnoses potential issues in our DQN implementation.")
    
    # Register ALE environments if needed
    if 'ale_py' in globals() and hasattr(ale_py, '__version__'):
        gym.register_envs(ale_py)
    
    # Test with both regular and reduced action spaces
    for reduced_actions in [False, True]:
        action_mode = "Reduced" if reduced_actions else "Standard"
        print(f"\n{'-' * 40}")
        print(f"TESTING WITH {action_mode.upper()} ACTION SPACE")
        print(f"{'-' * 40}")
        
        # Create environment
        env = make_diagnostic_env("PongNoFrameskip-v4", reduced_actions=reduced_actions)
        
        # Create agent with the proper state shape
        state_shape = (4, 84, 84)  # Channels-first shape for PyTorch
        action_space_size = env.action_space.n
        print(f"Environment observation space: {env.observation_space}")
        print(f"Environment action space: {env.action_space}")
        
        # Set up agent with fast exploration decay for testing
        agent = DQNAgent(
            state_shape=state_shape,
            action_space=action_space_size,
            epsilon_decay_frames=50000  # Fast decay for testing
        )
        
        # Run diagnostic episodes
        print("\nRunning episode with high exploration...")
        explore_reward, explore_actions, explore_rewards, explore_q_values = run_diagnostic_episode(
            env, agent, explore=True, num_steps=1000
        )
        
        print("\nRunning episode with no exploration...")
        agent.current_frames = 100000  # Force low epsilon
        no_explore_reward, no_explore_actions, no_explore_rewards, no_explore_q_values = run_diagnostic_episode(
            env, agent, explore=False, num_steps=1000
        )
        
        # Clean up
        env.close()
        
        print(f"\n{action_mode} Action Space Results:")
        print(f"Explore mode reward: {explore_reward}")
        print(f"No explore mode reward: {no_explore_reward}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("Check visualizations in the data/pong/diagnostics directory")
    print("=" * 80)

if __name__ == "__main__":
    main()
