"""
Training and rollout collection functions for Pong PPO implementation.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import json
import datetime
from collections import deque

from pong_ppo_minimal_env import device

# ANSI color codes for terminal output
class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# Log prefixes with colors
GAME_PREFIX = f"{Color.GREEN}[GAME]{Color.END}"
PPO_PREFIX = f"{Color.BLUE}[PPO]{Color.END}"
EVAL_PREFIX = f"{Color.PURPLE}[EVAL]{Color.END}"
PROGRESS_PREFIX = f"{Color.YELLOW}[PROGRESS]{Color.END}"
DIAGNOSTICS_PREFIX = f"{Color.CYAN}[DIAGNOSTICS]{Color.END}"

# Global tracking variables
global_episode_count = 0
global_step_count = 0

def collect_episodes(env, agent, ppo, num_episodes=10, render=False, debug=True):
    """
    Collect a specified number of complete episodes for PPO training with detailed diagnostics.
    
    Args:
        env: Gym environment
        agent: Actor-critic agent
        ppo: PPO algorithm object
        num_episodes: Number of complete episodes to collect
        render: Whether to render the environment during collection
        debug: Whether to include extra diagnostics
        
    Returns:
        rollout: Dictionary of collected experience
        episode_rewards: List of rewards for each completed episode
        episode_lengths: List of lengths for each completed episode
    """
    global global_episode_count, global_step_count
    
    # Storage for the rollout
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    entropies = []
    action_probs_list = []
    
    # For tracking episodes
    episode_rewards = []
    episode_lengths = []
    batch_episode_count = 0
    
    # Detailed episode data for JSON
    episode_data = []
    
    print(f"{GAME_PREFIX} Collecting {num_episodes} complete episodes...")
    
    # Continue until we've collected the desired number of episodes
    while batch_episode_count < num_episodes:
        # Start a new episode
        episode_reward = 0
        episode_length = 0
        
        # Update global counters for detailed logging
        global_episode_count += 1
        local_episode_number = global_episode_count
        
        # Reset environment with random seed
        state, _ = env.reset(seed=np.random.randint(0, 1000000))
        state_tensor = torch.FloatTensor(state).to(device)
        done = False
        
        # Run episode until completion
        while not done:
            # Get action, log prob, value, and entropy
            with torch.no_grad():
                # Apply higher temperature sampling for exploration in early episodes
                if batch_episode_count < 0.1 * num_episodes:
                    if state_tensor.dim() == 3:
                        state_tensor = state_tensor.unsqueeze(0)
                    
                    logits, state_value = agent(state_tensor)
                    logits = logits / 1.2  # Temperature > 1 = more exploration
                    action_probs = F.softmax(logits, dim=1)
                    dist = torch.distributions.Categorical(action_probs)
                    action_tensor = dist.sample()
                    log_prob = dist.log_prob(action_tensor).item()
                    action = action_tensor.item()
                    value = state_value.item()
                    entropy = dist.entropy().item()
                    action_probs = action_probs.squeeze().cpu().numpy()
                else:
                    # Regular sampling
                    action, log_prob, value, entropy, action_probs = agent.get_action(state_tensor)
            
            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update episode tracking
            episode_reward += reward
            episode_length += 1
            
            # Store the transition with safety checks for numerical stability
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            
            # Safety checks for numerical stability
            if np.isnan(log_prob) or np.isinf(log_prob):
                print(f"Warning: NaN/Inf detected in log_prob, using -1.0 instead")
                log_prob = -1.0
                
            if np.isnan(value) or np.isinf(value):
                print(f"Warning: NaN/Inf detected in value, using 0.0 instead")
                value = 0.0
                
            if np.isnan(entropy) or np.isinf(entropy):
                print(f"Warning: NaN/Inf detected in entropy, using 0.5 instead")
                entropy = 0.5
            
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)
            action_probs_list.append(action_probs)
            
            # Save diagnostic info about action distribution periodically
            if debug and episode_length % 200 == 0:
                with open("data/pong/diagnostics/action_selection.txt", "a") as f:
                    f.write(f"Episode {local_episode_number}, Step {episode_length}: Action={action}, " +
                           f"Probs=[{action_probs[0]:.4f}, {action_probs[1]:.4f}, {action_probs[2]:.4f}], " +
                           f"Entropy={entropy:.4f}\n")
            
            # Update state for next step
            state = next_state
            state_tensor = torch.FloatTensor(state).to(device)
            
        # Update global step count
        global_step_count += episode_length
        
        # Episode complete - log immediately for each episode
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        batch_episode_count += 1
        
        # Create episode data entry for JSON
        episode_data_entry = {
            'episode_id': local_episode_number,
            'reward': episode_reward,
            'length': episode_length,
            'timestamp': time.time(),
            'total_steps': global_step_count
        }
        episode_data.append(episode_data_entry)
        
        # Log episode stats - print immediately for better tracking
        print(f"{GAME_PREFIX} Episode {local_episode_number} complete: reward={episode_reward:.1f}, length={episode_length}")
        print(f"{PROGRESS_PREFIX} Total progress: Episodes={global_episode_count}, Steps={global_step_count}")
        
        # Save episode statistics
        if debug:
            with open("data/pong/diagnostics/episode_detailed_stats.txt", "a") as f:
                f.write(f"Episode {local_episode_number}: reward={episode_reward:.1f}, length={episode_length}, " +
                       f"total_steps={global_step_count}\n")
            
            # No longer saving individual episode JSONs - all episodes will be in a single file
    
    # Get value of final state (for GAE calculation)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).to(device)
        _, _, next_value, _, _ = agent.get_action(state_tensor)
    
    # Calculate advantages and returns with diagnostics enabled
    advantages, returns = ppo.compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        debug=debug
    )
    
    # Print rollout summary
    total_steps = len(states)
    print(f"Collection complete: {total_steps} total steps across {num_episodes} episodes")
    print(f"Episode rewards: mean={np.mean(episode_rewards):.1f}, min={np.min(episode_rewards):.1f}, max={np.max(episode_rewards):.1f}")
    print(f"Episode lengths: mean={np.mean(episode_lengths):.1f}, min={np.min(episode_lengths):.1f}, max={np.max(episode_lengths):.1f}")
    
    # Prepare rollout data
    rollout = {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'old_log_probs': np.array(log_probs),
        'values': np.array(values),
        'advantages': advantages,
        'returns': returns,
        'entropies': np.array(entropies),
        'action_probs': np.array(action_probs_list)
    }
    
    # Save additional diagnostic information - based on update count
    if debug:
        # Determine visualization trigger points
        update_count = getattr(ppo, 'update_count', 0)
        should_save_vis = (update_count == 0 or update_count % 12 == 0)  # Same interval as other diagnostics
        
        if should_save_vis:
            # Visualize rewards distribution
            plt.figure(figsize=(10, 6))
            plt.hist(rewards, bins=20)
            plt.title('Reward Distribution in Episodes')
            plt.xlabel('Reward')
            plt.ylabel('Count')
            plt.savefig(f"data/pong/diagnostics/reward_distribution_update_{update_count}.png")
            plt.close()
            
            # Visualize episode rewards
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(episode_rewards)), episode_rewards)
            plt.axhline(y=np.mean(episode_rewards), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(episode_rewards):.1f}')
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.legend()
            plt.savefig(f"data/pong/diagnostics/episode_rewards_update_{update_count}.png")
            plt.close()
        
        # Always log text statistics - they don't take much space
        with open("data/pong/diagnostics/rollout_stats.txt", "a") as f:
            f.write(f"Collection at update {update_count}:\n")
            f.write(f"  Total Steps: {total_steps}\n")
            f.write(f"  Episodes: {num_episodes}\n")
            f.write(f"  Episode Rewards: mean={np.mean(episode_rewards):.1f}, min={np.min(episode_rewards):.1f}, max={np.max(episode_rewards):.1f}\n")
            f.write(f"  Episode Lengths: mean={np.mean(episode_lengths):.1f}, min={np.min(episode_lengths):.1f}, max={np.max(episode_lengths):.1f}\n")
            f.write(f"  Rewards: mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}\n")
            f.write(f"  Values: mean={np.mean(values):.4f}, std={np.std(values):.4f}\n")
            f.write(f"  Advantages: mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}\n")
            f.write(f"  Returns: mean={np.mean(returns):.4f}, std={np.std(returns):.4f}\n")
            f.write(f"  Entropies: mean={np.mean(entropies):.4f}, std={np.std(entropies):.4f}\n\n")
    
    return rollout, episode_rewards, episode_lengths


def evaluate_agent(env, agent, num_episodes=5, render=False, seed=None, current_timesteps=0):
    """
    Evaluate the agent on several episodes.
    
    Args:
        env: Gym environment
        agent: Actor-critic agent
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        seed: Random seed for reproducibility
        current_timesteps: Current training timesteps for visualization naming
        
    Returns:
        avg_reward: Average episode reward
        avg_length: Average episode length
        all_ep_rewards: List of all episode rewards
        action_counts: Dictionary of action counts
    """
    episode_rewards = []
    episode_lengths = []
    action_counts = {0: 0, 1: 0, 2: 0}  # NOOP, UP, DOWN
    all_entropies = []
    
    print(f"Evaluating agent over {num_episodes} episodes...")
    
    for episode_idx in range(num_episodes):
        # Set seed with offset for each episode to ensure diversity
        if seed is not None:
            episode_seed = seed + episode_idx
        else:
            episode_seed = np.random.randint(0, 1000000)
        
        state, _ = env.reset(seed=episode_seed)
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Choose action based on policy
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action, _, _, entropy, action_probs = agent.get_action(state_tensor, deterministic=False)
            
            # Track action distribution
            action_counts[action] += 1
            all_entropies.append(entropy)
            
            # Take a step in the environment
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode_idx+1}: reward={episode_reward:.1f}, length={episode_length}")
    
    # Calculate statistics
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    total_actions = sum(action_counts.values())
    action_distribution = {k: v/total_actions for k, v in action_counts.items()}
    avg_entropy = np.mean(all_entropies)
    
    # Print evaluation results
    print(f"Evaluation complete:")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average length: {avg_length:.2f}")
    print(f"  Action distribution: NOOP={action_distribution[0]:.2f}, UP={action_distribution[1]:.2f}, DOWN={action_distribution[2]:.2f}")
    print(f"  Average entropy: {avg_entropy:.4f}")
    
    # Save evaluation results
    with open("data/pong/diagnostics/evaluation_results.txt", "a") as f:
        f.write(f"Evaluation Results:\n")
        f.write(f"  Episodes: {num_episodes}\n")
        f.write(f"  Average reward: {avg_reward:.2f}\n")
        f.write(f"  Average length: {avg_length:.2f}\n")
        f.write(f"  Action distribution: NOOP={action_distribution[0]:.2f}, UP={action_distribution[1]:.2f}, DOWN={action_distribution[2]:.2f}\n")
        f.write(f"  Average entropy: {avg_entropy:.4f}\n")
        f.write(f"  All rewards: {episode_rewards}\n\n")
    
    # Only create evaluation plot at certain intervals based on total timesteps
    if current_timesteps == 0 or current_timesteps % 50000 < 5000:  # At start or approximately every 50k steps
        # Plot reward distribution
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(episode_rewards)), episode_rewards)
        plt.axhline(y=avg_reward, color='r', linestyle='--', label=f'Mean: {avg_reward:.2f}')
        plt.title('Evaluation Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(f"data/pong/diagnostics/eval_rewards_{current_timesteps}.png")
        plt.close()
    
    return avg_reward, avg_length, episode_rewards, action_counts


def train(env, eval_env, agent, ppo, total_timesteps=1_000_000, seed=None, rollout_steps=128,
          eval_freq=10, save_freq=50, log_freq=1, optimizer_state=None, 
          start_timesteps=0, start_updates=0, start_episodes=0, loaded_episode_records=None):
    """
    Main training function for PPO on Pong with comprehensive diagnostics.
    
    Args:
        env: Training environment
        eval_env: Evaluation environment
        agent: Actor-critic agent
        ppo: PPO algorithm object
        total_timesteps: Total timesteps for training
        seed: Random seed
        rollout_steps: Number of steps per rollout
        eval_freq: Evaluation frequency (in updates)
        save_freq: Model saving frequency (in updates)
        log_freq: Logging frequency (in updates)
        optimizer_state: Optional optimizer state to load for continued training
        start_timesteps: Starting timestep count for resumed training
        start_updates: Starting update count for resumed training
        start_episodes: Starting episode count for resumed training
        
    Returns:
        agent: Trained agent
        episode_records: List of episode record dictionaries
    """
    # Set up logging directory
    log_dir = "data/pong"
    os.makedirs(log_dir, exist_ok=True)
    diagnostics_dir = os.path.join(log_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Set random seeds
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    
    # Variables for tracking training progress
    total_timesteps_so_far = start_timesteps
    updates = start_updates
    best_eval_reward = float('-inf')
    training_start_time = time.time()
    
    # Training data storage for JSON - using episode ID as primary key for human readability
    episode_records = loaded_episode_records if loaded_episode_records else {}  # Dictionary with episode numbers as keys
    eval_records = {}     # Dictionary with update numbers as keys
    
    # Episode tracking variables
    episode_count = start_episodes
    episode_reward = 0
    episode_length = 0
    
    # If resuming, load optimizer state
    if optimizer_state is not None:
        ppo.optimizer.load_state_dict(optimizer_state)
        print(f"{PROGRESS_PREFIX} Resuming from update {updates}, episode {episode_count}, timestep {total_timesteps_so_far}")
    
    # Create log files
    with open(os.path.join(diagnostics_dir, "training_log.txt"), "w") as f:
        f.write("PPO Training Log\n")
        f.write("----------------\n\n")
        f.write(f"Device: {device}\n")
        f.write(f"Total timesteps: {total_timesteps}\n")
        f.write(f"Rollout steps: {rollout_steps}\n\n")
    
    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Updates needed: ~{total_timesteps // rollout_steps}")
    
    # Main training loop
    while total_timesteps_so_far < total_timesteps:
        # Determine number of episodes to collect based on progress
        # Start with more episodes early on, then adjust based on typical episode length
        if episode_count < 10:
            num_episodes = 10  # More episodes at the beginning for better initial data
        else:
            # Calculate average episode length from recent episodes using dictionary
            sorted_keys = sorted(episode_records.keys(), key=lambda x: int(x))
            recent_keys = sorted_keys[-10:] if len(sorted_keys) >= 10 else sorted_keys
            recent_lengths = [episode_records[key]['length'] for key in recent_keys]
            avg_ep_length = np.mean(recent_lengths) if recent_lengths else 1000
            
            # Adjust episode count to roughly get rollout_steps number of steps
            # But ensure at least 5 episodes for stability
            num_episodes = max(5, min(20, int(rollout_steps / max(1, avg_ep_length))))
        
        print(f"{PROGRESS_PREFIX} Starting update {updates+1} with {num_episodes} episodes...")
        
        # Collect experience by complete episodes
        rollout, ep_rewards, ep_lengths = collect_episodes(
            env=env,
            agent=agent,
            ppo=ppo,
            num_episodes=num_episodes,
            debug=(updates % 5 == 0)  # More detailed debugging every 5 updates
        )
        
        # Get total steps collected in this batch
        steps_collected = sum(ep_lengths)
        
        # Update agent using PPO with dynamic batch size based on collected data
        # Use larger minibatch size for stability but scale with collected data size
        minibatch_size = min(128, max(64, steps_collected // 16))
        update_metrics = ppo.update(
            rollout=rollout,
            n_epochs=8,  # 8 epochs for thorough training
            batch_size=minibatch_size
        )
        
        # Update training progress
        total_timesteps_so_far += steps_collected
        updates += 1
        
        # Process completed episodes (all episodes are complete in our new approach)
        for i, (ep_reward, ep_length) in enumerate(zip(ep_rewards, ep_lengths)):
            # Increment episode counter
            episode_count += 1
            
            # Create detailed episode record with all metrics
            episode_record = {
                'episode_id': episode_count,
                'global_episode_id': global_episode_count - len(ep_rewards) + i + 1,
                'reward': ep_reward,
                'length': ep_length,
                'policy_loss': update_metrics['policy_loss'],
                'value_loss': update_metrics['value_loss'],
                'entropy': update_metrics['entropy'],
                'grad_norm': update_metrics.get('grad_norm', 0),
                'timesteps': total_timesteps_so_far,
                'timestamp': time.time(),
                'elapsed_time': time.time() - training_start_time,
                'update_number': updates
            }
            
            # Add to episode records dictionary using episode_id as the key for human readability
            episode_records[str(episode_count)] = episode_record
            
            # Log episode
            if episode_count % log_freq == 0 or ep_reward > -15:  # Log more often for good episodes
                elapsed_time = time.time() - training_start_time
                print(f"{PPO_PREFIX} Episode {episode_count}, " +
                      f"Timesteps: {total_timesteps_so_far}/{total_timesteps} ({total_timesteps_so_far/total_timesteps*100:.1f}%), " + 
                      f"Reward: {ep_reward:.1f}, " +
                      f"Length: {ep_length}")
                print(f"{DIAGNOSTICS_PREFIX} Policy Loss: {update_metrics['policy_loss']:.6f}, " +
                      f"Value Loss: {update_metrics['value_loss']:.6f}, " +
                      f"Entropy: {update_metrics['entropy']:.6f}, " +
                      f"Time: {elapsed_time:.2f}s")
                
                # Append to log file
                with open(os.path.join(diagnostics_dir, "training_log.txt"), "a") as f:
                    f.write(f"Episode {episode_count}, " +
                           f"Timesteps: {total_timesteps_so_far}, " +
                           f"Reward: {ep_reward:.1f}, " +
                           f"Length: {ep_length}, " +
                           f"Policy Loss: {update_metrics['policy_loss']:.6f}, " +
                           f"Value Loss: {update_metrics['value_loss']:.6f}, " +
                           f"Entropy: {update_metrics['entropy']:.6f}, " +
                           f"Time: {elapsed_time:.2f}s\n")
                
            # Save episode records to JSON after each batch
            if i == len(ep_rewards) - 1:
                with open(os.path.join(log_dir, "training_data.json"), "w") as f:
                    json.dump(episode_records, f, indent=2)
                
                # Generate training trends chart every 10 episodes (not tied to diagnostic mode)
                if episode_count % 10 == 0:
                    visualize_training_trends(episode_records, os.path.join(log_dir, "training_data.png"))
        
        # Evaluate agent
        if updates % eval_freq == 0:
            print(f"{EVAL_PREFIX} Running evaluation after {total_timesteps_so_far} steps...")
            eval_reward, eval_length, eval_episodes, _ = evaluate_agent(
                env=eval_env,
                agent=agent,
                num_episodes=5,
                current_timesteps=total_timesteps_so_far
            )
            
            # Create eval record
            eval_record = {
                'update': updates,
                'timesteps': total_timesteps_so_far,
                'avg_reward': eval_reward,
                'avg_length': eval_length,
                'episode_rewards': eval_episodes,
                'timestamp': time.time(),
                'elapsed_time': time.time() - training_start_time
            }
            
            # Add to eval records dictionary using update number as the key
            eval_records[str(updates)] = eval_record
            
            # Save evaluation data
            with open(os.path.join(log_dir, "evaluation_data.json"), "w") as f:
                json.dump(eval_records, f, indent=2)
            
            # Save best model
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(agent.state_dict(), os.path.join(models_dir, "best_model.pt"))
                print(f"{EVAL_PREFIX} New best model saved with reward: {best_eval_reward:.1f}")
        
        # Save checkpoint
        if updates % save_freq == 0:
            # Get diagnostic counter for continuous activation visualization
            diagnostic_counter = agent.get_diagnostic_counter()
            
            torch.save({
                'agent': agent.state_dict(),
                'optimizer': ppo.optimizer.state_dict(),
                'update': updates,
                'timesteps': total_timesteps_so_far,
                'episode': episode_count,
                'diagnostic_counter': diagnostic_counter  # Save for visualization continuity
            }, os.path.join(models_dir, f"checkpoint_{updates}.pt"))
            print(f"Checkpoint saved at update {updates} (diagnostic counter: {diagnostic_counter})")
            
            # Save training progress plot
            visualize_training_progress(episode_records, os.path.join(log_dir, "pong_ppo_minimal_progress.png"))
    
    # Final save
    torch.save(agent.state_dict(), os.path.join(models_dir, "final_model.pt"))
    print(f"Training complete. Final model saved.")
    
    # Final evaluation
    eval_reward, eval_length, _, _ = evaluate_agent(
        env=eval_env,
        agent=agent,
        num_episodes=10,
        current_timesteps=total_timesteps_so_far
    )
    print(f"Final evaluation: Average Reward: {eval_reward:.1f}")
    
    # Save final training data and plot
    with open(os.path.join(log_dir, "training_data.json"), "w") as f:
        json.dump(episode_records, f, indent=2)
    
    visualize_training_progress(episode_records, os.path.join(log_dir, "pong_ppo_minimal_progress.png"))
    
    return agent, episode_records


def visualize_training_progress(episode_records, filename="pong_ppo_minimal_progress.png"):
    """
    Visualize and save training progress as a single chart for rewards
    and separate charts for metrics.
    
    Args:
        episode_records: Dictionary with episode IDs as keys and record data as values
        filename: Output file name
    """
    if len(episode_records) == 0:
        return  # Nothing to plot yet
    
    # Create figure for reward plot
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title('PPO Training Progress on Pong', fontsize=16)
    
    # Sort episode records by episode ID (numerical order)
    sorted_keys = sorted(episode_records.keys(), key=lambda x: int(x))
    sorted_records = [episode_records[key] for key in sorted_keys]
    
    # Extract data from episode records
    episodes = [record['episode_id'] for record in sorted_records]
    rewards = [record['reward'] for record in sorted_records]
    
    # Handle NaN values and convert to appropriate data types for the metrics
    policy_losses = []
    value_losses = []
    entropies = []
    
    for record in sorted_records:
        # Replace NaN or inf values with zeros to prevent plotting issues
        policy_loss = record['policy_loss']
        policy_losses.append(0.0 if np.isnan(policy_loss) or np.isinf(policy_loss) else float(policy_loss))
        
        value_loss = record['value_loss']
        value_losses.append(0.0 if np.isnan(value_loss) or np.isinf(value_loss) else float(value_loss))
        
        entropy = record['entropy']
        entropies.append(0.0 if np.isnan(entropy) or np.isinf(entropy) else float(entropy))
    
    # Plot rewards
    ax.plot(episodes, rewards, 'b-', linewidth=1.5, label='Episode Reward')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3)
    
    # Calculate 100-episode rolling average
    rolling_rewards = []
    window_size = 100
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        window = rewards[start_idx:i+1]
        rolling_rewards.append(np.mean(window))
    
    # Plot rolling average as bold red line (on top)
    ax.plot(episodes, rolling_rewards, 'r-', linewidth=3, label='100-Episode Avg', zorder=10)
    
    # Add legend
    ax.legend(loc='best')
    
    # Save the rewards plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    
    # Create a second figure for metrics
    fig2, (ax_policy, ax_value, ax_entropy) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig2.suptitle('PPO Training Metrics', fontsize=16)
    
    # Plot each metric on its own axis with appropriate scaling
    ax_policy.plot(episodes, policy_losses, 'g-', linewidth=2)
    ax_policy.set_ylabel('Policy Loss', fontsize=12, color='g')
    ax_policy.tick_params(axis='y', labelcolor='g')
    ax_policy.grid(True, alpha=0.3)
    
    ax_value.plot(episodes, value_losses, 'r-', linewidth=2)
    ax_value.set_ylabel('Value Loss', fontsize=12, color='r')
    ax_value.tick_params(axis='y', labelcolor='r')
    ax_value.grid(True, alpha=0.3)
    
    ax_entropy.plot(episodes, entropies, 'c-', linewidth=2)
    ax_entropy.set_ylabel('Entropy', fontsize=12, color='c')
    ax_entropy.set_xlabel('Episode', fontsize=12)
    ax_entropy.tick_params(axis='y', labelcolor='c')
    ax_entropy.grid(True, alpha=0.3)
    
    # Adjust layout and save the metrics plot
    plt.tight_layout()
    plt.savefig(filename.replace('.png', '_metrics.png'))
    plt.close(fig2)


def visualize_training_trends(episode_records, filename="training_data.png"):
    """
    Generate chart showing reward trend by episode number with a 100-episode rolling average.
    This chart is always generated (not just in diagnostic mode).
    
    Args:
        episode_records: Dictionary with episode IDs as keys and record data as values
        filename: Output file name
    """
    if len(episode_records) == 0:
        return  # Nothing to plot yet
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title('Pong Training Progress', fontsize=16)
    
    # Sort episodes
    sorted_keys = sorted(episode_records.keys(), key=lambda x: int(x))
    sorted_records = [episode_records[key] for key in sorted_keys]
    
    # Extract data
    episodes = [record['episode_id'] for record in sorted_records]
    rewards = [record['reward'] for record in sorted_records]
    
    # Plot rewards
    ax.plot(episodes, rewards, 'b-', linewidth=1.5, label='Episode Reward')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12, color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3)
    
    # Calculate 100-episode rolling average
    rolling_rewards = []
    window_size = 100
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        window = rewards[start_idx:i+1]
        rolling_rewards.append(np.mean(window))
    
    # Plot rolling average as bold red line (on top)
    ax.plot(episodes, rolling_rewards, 'r-', linewidth=3, label='100-Episode Avg', zorder=10)
    
    # Add legend
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
