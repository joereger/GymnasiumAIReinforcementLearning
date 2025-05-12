"""
PPO training script for Pong.
Implements training loop, rollout collection, and evaluation.
"""

import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque

from pong_env_wrappers import make_pong_env, device
from pong_ppo_model import PPOActorCritic, PPO

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Pong")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total timesteps for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--rollout-steps", type=int, default=128,
                        help="Number of steps per rollout")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluation frequency (in timesteps)")
    parser.add_argument("--save-freq", type=int, default=100_000,
                        help="Model saving frequency (in timesteps)")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
    parser.add_argument("--log-freq", type=int, default=1000,
                        help="Logging frequency (in timesteps)")
    parser.add_argument("--lr", type=float, default=2.5e-4,
                        help="Learning rate")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to load model from")
    
    return parser.parse_args()

def collect_rollout(env, agent, ppo, rollout_steps, render=False, seed=None):
    """
    Collect experience rollout for PPO training.
    
    Args:
        env: Gym environment
        agent: Actor-critic agent
        ppo: PPO algorithm object
        rollout_steps: Number of steps to collect
        render: Whether to render the environment
        
    Returns:
        rollout: Dictionary of collected experience
    """
    # Storage for the rollout
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    
    # Initial state (pass seed to reset if needed)
    state, _ = env.reset(seed=None if seed is None else np.random.randint(0, 1000000))
    state_tensor = torch.FloatTensor(state).to(device)
    
    # Collect rollout
    for _ in range(rollout_steps):
        # Get action, log prob, and value
        with torch.no_grad():
            action, log_prob, value = agent.get_action(state_tensor)
        
        # Take a step in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store the transition
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(float(done))
        log_probs.append(log_prob)
        values.append(value)
        
        # Update state
        state = next_state
        state_tensor = torch.FloatTensor(state).to(device)
        
        # Reset if episode is done
        if done:
            state, _ = env.reset(seed=np.random.randint(0, 1000000))
            state_tensor = torch.FloatTensor(state).to(device)
    
    # Get value of final state (for GAE calculation)
    with torch.no_grad():
        _, _, next_value = agent.get_action(state_tensor)
    
    # Calculate advantages and returns
    advantages, returns = ppo.compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value
    )
    
    # Prepare rollout data
    rollout = {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'old_log_probs': np.array(log_probs),
        'values': np.array(values),
        'advantages': advantages,
        'returns': returns
    }
    
    return rollout

def evaluate_agent(env, agent, num_episodes=5, render=False, epsilon=0.0):
    """
    Evaluate the agent on several episodes.
    
    Args:
        env: Gym environment
        agent: Actor-critic agent
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        epsilon: Probability of random action for exploration
        
    Returns:
        avg_reward: Average episode reward
        avg_length: Average episode length
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode_idx in range(num_episodes):
        state, _ = env.reset(seed=np.random.randint(0, 1000000))
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Choose action based on policy with epsilon-greedy exploration
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action, _, _ = agent.get_action(state_tensor, deterministic=True)
            
            # Take a step in the environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    return avg_reward, avg_length

def visualize_training_progress(timesteps, rewards, eval_freq, filename="pong_ppo_training_progress.png"):
    """
    Visualize and save training progress as a plot.
    
    Args:
        timesteps: List of timesteps
        rewards: List of corresponding rewards
        eval_freq: Evaluation frequency (for x-axis)
        filename: Output file name
    """
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, rewards, 'b-', linewidth=2)
    plt.title('PPO Training Progress on Pong')
    plt.xlabel('Timesteps')
    plt.ylabel('Average Episode Reward')
    plt.grid(True)
    
    # Save the plot
    os.makedirs('data/pong', exist_ok=True)
    plt.savefig(os.path.join('data/pong', filename))
    plt.close()

def train_ppo():
    """Main training function for PPO on Pong."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging directory
    log_dir = os.path.join('data', 'pong')
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(args.seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Create environment using robust environment creation function
    env = make_pong_env(reduced_actions=True, seed=args.seed)
    eval_env = make_pong_env(reduced_actions=True, seed=args.seed + 1000)
    
    # Get environment info
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Using device: {device}")
    
    # Create actor-critic agent
    agent = PPOActorCritic(
        input_channels=obs_shape[0],
        action_dim=n_actions
    ).to(device)
    
    # Load model if specified
    if args.load_model is not None:
        agent.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded model from {args.load_model}")
    
    # Create PPO algorithm
    ppo = PPO(
        actor_critic=agent,
        learning_rate=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Set up tracking variables
    timesteps = []
    eval_rewards = []
    episode_rewards = []
    
    # For tracking recent rewards
    recent_rewards = deque(maxlen=10)
    current_episode_reward = 0
    
    # Start timer
    start_time = time.time()
    
    # Main training loop
    total_timesteps = 0
    updates = 0
    episodes = 0
    
    print("Starting PPO training...")
    
    while total_timesteps < args.total_timesteps:
        # Collect rollout
        rollout = collect_rollout(
            env=env,
            agent=agent,
            ppo=ppo,
            rollout_steps=args.rollout_steps
        )
        
        # Update agent
        loss_metrics = ppo.update(
            rollout=rollout,
            n_epochs=4,
            batch_size=64
        )
        
        # Update tracking variables
        total_timesteps += args.rollout_steps
        updates += 1
        
        # Track episode completion
        for i, done in enumerate(rollout['dones']):
            current_episode_reward += rollout['rewards'][i]
            if done:
                episodes += 1
                recent_rewards.append(current_episode_reward)
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
        
        # Logging
        if updates % (args.log_freq // args.rollout_steps) == 0:
            # Calculate time statistics
            elapsed_time = time.time() - start_time
            fps = total_timesteps / elapsed_time
            
            if recent_rewards:
                avg_recent_reward = np.mean(recent_rewards)
            else:
                avg_recent_reward = 0
            
            print(f"Update: {updates}, Timesteps: {total_timesteps}, Episodes: {episodes}")
            print(f"Recent average reward: {avg_recent_reward:.2f}")
            print(f"Policy loss: {loss_metrics['policy_loss']:.4f}, Value loss: {loss_metrics['value_loss']:.4f}")
            print(f"Entropy: {loss_metrics['entropy']:.4f}, FPS: {fps:.1f}")
            print("-------------------------------------")
        
        # Evaluate agent
        if total_timesteps % args.eval_freq == 0 or total_timesteps >= args.total_timesteps:
            avg_reward, avg_length = evaluate_agent(
                env=eval_env,
                agent=agent,
                num_episodes=5,
                render=args.render,
                epsilon=0.05  # Small exploration for more realistic evaluation
            )
            
            print(f"\nEVALUATION at timestep {total_timesteps}:")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Average episode length: {avg_length:.2f}")
            print("-------------------------------------\n")
            
            # Save tracking variables for plotting
            timesteps.append(total_timesteps)
            eval_rewards.append(avg_reward)
            
            # Visualize training progress
            visualize_training_progress(timesteps, eval_rewards, args.eval_freq)
        
        # Save model
        if total_timesteps % args.save_freq == 0 or total_timesteps >= args.total_timesteps:
            model_path = os.path.join(model_dir, f"ppo_pong_{total_timesteps}.pt")
            torch.save(agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Final evaluation
    print("\nTraining completed. Final evaluation:")
    avg_reward, avg_length = evaluate_agent(
        env=eval_env,
        agent=agent,
        num_episodes=10,
        render=args.render
    )
    print(f"Final average reward over 10 episodes: {avg_reward:.2f}")
    
    # Save final model
    final_model_path = os.path.join(model_dir, "ppo_pong_final.pt")
    torch.save(agent.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Clean up
    env.close()
    eval_env.close()
    
    # Return final performance
    return avg_reward

if __name__ == "__main__":
    train_ppo()
