import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_training_data(stats, filename="pong_ppo_progress.png"):
    """
    Create a comprehensive visualization of PPO training progress.
    
    Plots multiple metrics on different axes:
    1. Episode rewards and rolling average
    2. Policy entropy
    3. Policy and value loss
    
    Args:
        stats: Dictionary containing training statistics
        filename: Path to save the plot
    """
    if not stats or "episode_rewards" not in stats or len(stats["episode_rewards"]) == 0:
        print("No stats to plot.")
        return

    # Extract data
    episodes = np.arange(1, len(stats["episode_rewards"]) + 1)
    rewards = np.array(stats["episode_rewards"])
    
    # Calculate rolling average reward (100 episodes)
    window_size = min(100, len(rewards))
    avg_rewards = np.zeros_like(rewards, dtype=float)
    for i in range(len(rewards)):
        avg_rewards[i] = np.mean(rewards[max(0, i-window_size+1):i+1])
    
    # Create multi-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle('Pong PPO Training Progress (Experiment 3)', fontsize=16)
    
    # Panel 1: Rewards
    ax1 = axes[0]
    color = 'tab:blue'
    ax1.set_ylabel('Reward', color='black')
    ax1.plot(episodes, rewards, label='Episode Reward', alpha=0.6, color=color, linewidth=1)
    ax1.plot(episodes, avg_rewards, label='Avg Reward (100ep)', color='red', linewidth=2)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add a horizontal line at reward 0 (winning vs losing)
    if min(rewards) < 0 and max(rewards) > 0:
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add legend
    ax1.legend(loc='upper left')
    
    # Panel 2: Entropy
    ax2 = axes[1]
    if "entropies" in stats and len(stats["entropies"]) > 0:
        # Ensure we have timestamps matching rewards
        entropy_indices = np.linspace(0, len(episodes)-1, len(stats["entropies"])).astype(int)
        entropy_episodes = episodes[entropy_indices]
        
        color = 'tab:green'
        ax2.set_ylabel('Entropy', color='black')
        ax2.plot(entropy_episodes, stats["entropies"], label='Policy Entropy', color=color, linewidth=1.5)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')
    
    # Panel 3: Policy and Value Loss
    ax3 = axes[2]
    if "policy_losses" in stats and "value_losses" in stats and len(stats["policy_losses"]) > 0:
        # Ensure we have timestamps matching rewards
        loss_indices = np.linspace(0, len(episodes)-1, len(stats["policy_losses"])).astype(int)
        loss_episodes = episodes[loss_indices]
        
        # Two y-axes for different scale losses
        color1 = 'tab:red'
        color2 = 'tab:purple'
        
        # Policy loss
        ax3.set_ylabel('Policy Loss', color='black')
        ln1 = ax3.plot(loss_episodes, stats["policy_losses"], label='Policy Loss', color=color1, linewidth=1.5)
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Value loss on secondary y-axis
        ax3_twin = ax3.twinx()
        ax3_twin.set_ylabel('Value Loss', color='black')
        ln2 = ax3_twin.plot(loss_episodes, stats["value_losses"], label='Value Loss', color=color2, linewidth=1.5, linestyle=':')
        
        # Add combined legend
        lns = ln1 + ln2
        labels = [l.get_label() for l in lns]
        ax3.legend(lns, labels, loc='upper right')
    
    # Set the x-label on the bottom plot only
    ax3.set_xlabel('Episode')
    
    # Add experiment details in a text box
    textstr = '\n'.join([
        'PPO Experiment 3:',
        'LR: 3e-4',
        'Clip: 0.2',
        'Value Coef: 0.5',
        'Entropy Coef: 0.01',
        'GAE Lambda: 0.95',
        'PPO Epochs: 4'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.02, 0.05, textstr, transform=ax1.transaxes, fontsize=9,
            verticalalignment='bottom', bbox=props)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save and close
    plt.savefig(filename)
    plt.close(fig)
    print(f"Plot saved to {filename}")

def save_training_stats(stats, filename="pong_ppo_stats.json"):
    """
    Save training statistics to a JSON file.
    
    Args:
        stats: Dictionary containing training statistics
        filename: Path to save the stats
    """
    with open(filename, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"Stats saved to {filename}")

def load_training_stats(filename="pong_ppo_stats.json"):
    """
    Load training statistics from a JSON file.
    
    Args:
        filename: Path to load the stats from
        
    Returns:
        Dictionary containing training statistics, or empty dict if file not found
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        print(f"Stats file {filename} not found.")
        return {}
