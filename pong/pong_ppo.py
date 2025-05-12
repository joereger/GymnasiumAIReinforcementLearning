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
    parser.add_argument("--total-timesteps", type=int, default=25_000_000,
                        help="Total timesteps for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--rollout-steps", type=int, default=128,
                        help="Number of steps per rollout")
    parser.add_argument("--eval-freq", type=int, default=500,
                        help="Evaluation frequency (in episodes)")
    parser.add_argument("--save-freq", type=int, default=100,
                        help="Model saving frequency (in episodes)")
    parser.add_argument("--vis-freq", type=int, default=10,
                        help="Visualization frequency (in episodes)")
    parser.add_argument("--json-save-freq", type=int, default=10,
                        help="JSON saving frequency (in episodes)")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during evaluation")
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
        render: Whether to render the environment during collection
        seed: Random seed for environment resets
        
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
        # Render if requested - for environments with render_mode="human"
        if render and hasattr(env, 'render'):
            env.render()
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

def format_time(seconds):
    """
    Format time in seconds to HH:MM:SS.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def visualize_training_progress(training_data, filename="pong_ppo_training_progress.png"):
    """
    Visualize and save training progress as a single chart with multiple axes.
    
    Args:
        training_data: Dictionary containing training metrics
        filename: Output file name
    """
    if len(training_data['episodes']) == 0:
        return  # Nothing to plot yet
    
    # Create figure with primary axis
    fig, ax1 = plt.figure(figsize=(15, 8)), plt.gca()
    plt.title('PPO Training Progress on Pong', fontsize=16)
    
    # Get episode data
    episodes = training_data['episodes']
    rewards = training_data['rewards']
    policy_losses = training_data['policy_losses']
    value_losses = training_data['value_losses']
    entropies = training_data['entropies']
    
    # Plot rewards on primary axis
    reward_line, = ax1.plot(episodes, rewards, 'b-', linewidth=2, label='Episode Reward')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary axis for losses
    ax2 = ax1.twinx()
    
    # Plot loss metrics on secondary axis
    policy_line, = ax2.plot(episodes, policy_losses, 'g-', linewidth=2, label='Policy Loss')
    value_line, = ax2.plot(episodes, value_losses, 'r-', linewidth=2, label='Value Loss')
    entropy_line, = ax2.plot(episodes, entropies, 'c-', linewidth=2, label='Entropy')
    
    ax2.set_ylabel('Loss / Entropy', fontsize=12, color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Add legend for all plots
    lines = [reward_line, policy_line, value_line, entropy_line]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels, loc='best', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('data/pong', exist_ok=True)
    plt.savefig(os.path.join('data/pong', filename))
    plt.close()

def train_ppo(env, eval_env, args, train_render=False):
    """
    Main training function for PPO on Pong.
    
    Args:
        env: Training environment
        eval_env: Evaluation environment
        args: Command line arguments
        train_render: Whether to render during training
    """
    import gc  # For garbage collection
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
    
    # Variables for tracking training progress and continuity
    total_timesteps = 0
    episodes = 0
    best_eval_reward = float('-inf')
    best_model_episode = 0
    training_start_time = time.time()
    
    # Variables to prevent duplicate operations
    last_saved_episode = 0
    last_json_saved_episode = 0
    last_visualized_episode = 0
    last_eval_episode = 0
    
    # Check for existing training data when loading a model
    existing_data = None
    if args.load_model is not None:
        # Load model weights
        agent.load_state_dict(torch.load(args.load_model, map_location=device))
        print(f"Loaded model from {args.load_model}")
        
        # Try to load existing training progress
        progress_path = os.path.join(log_dir, 'ppo_training_progress.json')
        if os.path.exists(progress_path):
            try:
                import json
                with open(progress_path, 'r') as f:
                    existing_data = json.load(f)
                    
                if len(existing_data) > 0:
                    # Resume episodes count and timesteps
                    episodes = existing_data[-1]['episode']
                    total_timesteps = existing_data[-1]['timesteps']
                    
                    # Extract best model info if it exists
                    best_rewards = [ep.get('eval_reward', float('-inf')) for ep in existing_data]
                    if any(r != float('-inf') for r in best_rewards):
                        best_idx = np.argmax([r if r != float('-inf') else float('-inf') for r in best_rewards])
                        best_eval_reward = best_rewards[best_idx]
                        best_model_episode = existing_data[best_idx]['episode']
                    
                    # Adjust start time to maintain cumulative time
                    if 'cumulative_time' in existing_data[-1]:
                        training_start_time = time.time() - existing_data[-1]['cumulative_time']
                    
                    print(f"Resuming training from episode {episodes}, timestep {total_timesteps}")
                    if best_eval_reward != float('-inf'):
                        print(f"Best model so far: {best_eval_reward:.2f} reward at episode {best_model_episode}")
            except Exception as e:
                print(f"Error loading training progress: {e}")
                print("Starting fresh training data")
    
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
    episode_lengths = []
    episode_times = []
    policy_losses = []
    value_losses = []
    entropies = []
    
    # For tracking data to save to JSON - initialize from existing data if available
    if existing_data:
        # Convert existing data from episode-based to arrays for internal tracking
        training_data = {
            'episodes': [ep['episode'] for ep in existing_data],
            'rewards': [ep['reward'] for ep in existing_data],
            'lengths': [ep['length'] for ep in existing_data],
            'times': [ep['time'] for ep in existing_data],
            'policy_losses': [ep['policy_loss'] for ep in existing_data],
            'value_losses': [ep['value_loss'] for ep in existing_data],
            'entropies': [ep['entropy'] for ep in existing_data],
            'cumulative_time': [ep['cumulative_time'] for ep in existing_data],
            'timesteps': [ep['timesteps'] for ep in existing_data],
            'eval_rewards': [ep.get('eval_reward', float('-inf')) for ep in existing_data]
        }
    else:
        # Start with empty tracking if no existing data
        training_data = {
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'times': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'cumulative_time': [],
            'timesteps': [],
            'eval_rewards': []
        }
    
    # For tracking current episode
    current_episode_reward = 0
    current_episode_length = 0
    episode_start_time = time.time()
    
    # Continuation of training loop
    updates = 0
    episode_policy_losses = []
    episode_value_losses = []
    episode_entropies = []
    
    print(f"PPO training running - will train until {args.total_timesteps:,} timesteps")
    
    while total_timesteps < args.total_timesteps:
        # Collect rollout, passing train_render
        rollout = collect_rollout(
            env=env,
            agent=agent,
            ppo=ppo,
            rollout_steps=args.rollout_steps,
            render=train_render
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
        
        # Track episode completion and update metrics
        for i, done in enumerate(rollout['dones']):
            current_episode_reward += rollout['rewards'][i]
            current_episode_length += 1
            episode_policy_losses.append(loss_metrics['policy_loss'])
            episode_value_losses.append(loss_metrics['value_loss'])
            episode_entropies.append(loss_metrics['entropy'])
            
            if done:
                episodes += 1
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                episode_end_time = time.time()
                episode_duration = episode_end_time - episode_start_time
                episode_times.append(episode_duration)
                cumulative_time = time.time() - training_start_time
                
                # Calculate average metrics for this episode
                avg_policy_loss = np.mean(episode_policy_losses)
                avg_value_loss = np.mean(episode_value_losses)
                avg_entropy = np.mean(episode_entropies)
                
                # Save metrics for this episode
                policy_losses.append(avg_policy_loss)
                value_losses.append(avg_value_loss)
                entropies.append(avg_entropy)
                
                # Add to JSON data
                training_data['episodes'].append(episodes)
                training_data['rewards'].append(float(current_episode_reward))
                training_data['lengths'].append(int(current_episode_length))
                training_data['times'].append(float(episode_duration))
                training_data['policy_losses'].append(float(avg_policy_loss))
                training_data['value_losses'].append(float(avg_value_loss))
                training_data['entropies'].append(float(avg_entropy))
                training_data['cumulative_time'].append(float(cumulative_time))
                training_data['timesteps'].append(int(total_timesteps))
                
                # Calculate cumulative time
                cumulative_time = time.time() - training_start_time
                
                # Log every episode completion - single line output
                fps = total_timesteps / cumulative_time
                formatted_time = format_time(cumulative_time)
                print(f"Episode {episodes:4d} | Reward: {current_episode_reward:6.1f} | Policy: {avg_policy_loss:.4f} | Value: {avg_value_loss:.4f} | Entropy: {avg_entropy:.4f} | Steps: {current_episode_length:4d} | Total: {total_timesteps:,} | Time: {formatted_time} | FPS: {fps:.1f}")
                
                # Save JSON data after each episode - organized by episode
                import json
                
                # Reorganize data by episode
                episode_data = []
                for i in range(len(training_data['episodes'])):
                    episode_data.append({
                        'episode': training_data['episodes'][i],
                        'reward': training_data['rewards'][i],
                        'length': training_data['lengths'][i],
                        'time': training_data['times'][i],
                        'policy_loss': training_data['policy_losses'][i],
                        'value_loss': training_data['value_losses'][i],
                        'entropy': training_data['entropies'][i],
                        'cumulative_time': training_data['cumulative_time'][i],
                        'timesteps': training_data['timesteps'][i]
                    })
                
                # Only save JSON every json_save_freq episodes
                if episodes % args.json_save_freq == 0 and episodes > last_json_saved_episode:
                    with open(os.path.join(log_dir, 'ppo_training_progress.json'), 'w') as f:
                        json.dump(episode_data, f, indent=2)
                    last_json_saved_episode = episodes
                
                # Visualize training progress every vis_freq episodes
                if episodes % args.vis_freq == 0 and episodes > last_visualized_episode:
                    visualize_training_progress(training_data)
                    last_visualized_episode = episodes
                
                # Reset episode tracking variables
                current_episode_reward = 0
                current_episode_length = 0
                episode_policy_losses = []
                episode_value_losses = []
                episode_entropies = []
                episode_start_time = time.time()
        
        # Evaluate agent periodically
        if episodes > 0 and ((episodes % args.eval_freq == 0 and episodes > last_eval_episode) or total_timesteps >= args.total_timesteps):
            last_eval_episode = episodes
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
            
            # Check if this is the best model so far
            is_best = False
            if avg_reward > best_eval_reward:
                best_eval_reward = avg_reward
                best_model_episode = episodes
                is_best = True
                
                # Save best model
                best_model_path = os.path.join(model_dir, "ppo_pong_best.pt")
                torch.save(agent.state_dict(), best_model_path)
                print(f"New best model saved with reward {avg_reward:.2f}!")
            
            print("-------------------------------------\n")
            
            # Save tracking variables for plotting
            timesteps.append(total_timesteps)
            eval_rewards.append(avg_reward)
            
            # Add evaluation results to JSON data
            training_data['eval_rewards'].append(float(avg_reward))
            
            # Update JSON with evaluation results
            import json
            episode_data = []
            for i in range(len(training_data['episodes'])):
                eval_reward = None
                if i < len(training_data['eval_rewards']):
                    eval_reward = training_data['eval_rewards'][i]
                
                episode_data.append({
                    'episode': training_data['episodes'][i],
                    'reward': training_data['rewards'][i],
                    'length': training_data['lengths'][i],
                    'time': training_data['times'][i],
                    'policy_loss': training_data['policy_losses'][i],
                    'value_loss': training_data['value_losses'][i],
                    'entropy': training_data['entropies'][i],
                    'cumulative_time': training_data['cumulative_time'][i],
                    'timesteps': training_data['timesteps'][i],
                    'eval_reward': eval_reward
                })
            
            with open(os.path.join(log_dir, 'ppo_training_progress.json'), 'w') as f:
                json.dump(episode_data, f, indent=2)
            
            # Force garbage collection to prevent memory build-up
            gc.collect()
        
        # Save model - only at exact save_freq intervals to prevent duplicate saves
        if episodes > 0 and episodes % args.save_freq == 0 and episodes > last_saved_episode:
            model_path = os.path.join(model_dir, f"ppo_pong_{total_timesteps}.pt")
            torch.save(agent.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            last_saved_episode = episodes
    
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

def prompt_user():
    """Prompt the user for rendering and model loading options."""
    # Ask about rendering during training
    train_render_choice = input("Do you want to view the game in human visible format during training? (y/n): ").lower()
    train_render = train_render_choice in ['y', 'yes']
    
    # Never render during evaluation
    eval_render = False
    
    # Ask about loading previous model
    load_model = None
    load_choice = input("Do you want to load a previous model to continue training? (y/n): ").lower()
    if load_choice in ['y', 'yes']:
        # Check for available models
        model_dir = os.path.join('data', 'pong', 'models')
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if model_files:
                model_files.sort()  # Sort to get newest models last
                print("\nAvailable models:")
                for i, model_file in enumerate(model_files):
                    print(f"{i+1}. {model_file}")
                
                # Let user select model
                selection = input(f"Enter model number (1-{len(model_files)}) or 'latest' for the most recent: ").lower()
                if selection == 'latest':
                    # Assuming the latest model is the last in the sorted list
                    load_model = os.path.join(model_dir, model_files[-1])
                else:
                    try:
                        idx = int(selection) - 1
                        if 0 <= idx < len(model_files):
                            load_model = os.path.join(model_dir, model_files[idx])
                        else:
                            print("Invalid selection. No model will be loaded.")
                    except ValueError:
                        print("Invalid input. No model will be loaded.")
            else:
                print("No model files found in data/pong/models/")
        else:
            print("Model directory not found.")
    
    return eval_render, train_render, load_model

if __name__ == "__main__":
    # Get command line arguments first
    args = parse_args()
    
    # Then override with user prompts
    eval_render, train_render, load_model = prompt_user()
    
    # Update args with user choices
    args.render = eval_render
    if load_model:
        args.load_model = load_model
        print(f"Will load model from: {load_model}")
    
    # Create training environment with render_mode based on user preference
    print(f"\nStarting training with rendering {'enabled' if train_render or eval_render else 'disabled'}")
    
    # Create environments with appropriate render_mode
    env_render_mode = "human" if train_render else None
    eval_env_render_mode = "human" if eval_render else None
    
    # Create environments
    env = make_pong_env(reduced_actions=True, seed=args.seed, render_mode=env_render_mode)
    eval_env = make_pong_env(reduced_actions=True, seed=args.seed + 1000, render_mode=eval_env_render_mode)
    
    # Pass environments to train_ppo
    train_ppo(env=env, eval_env=eval_env, args=args, train_render=train_render)
