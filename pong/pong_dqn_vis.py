import matplotlib.pyplot as plt
import numpy as np

def plot_training_data(episode_stats, filename="pong_training_progress.png"):
    """
    Create a visualization of training progress with multiple metrics.
    
    Plots:
    1. Episode rewards and 100-episode rolling average
    2. Average max Q-values per episode (if available)
    3. Average loss per episode (if available)
    
    Each metric is plotted with a different color and on its own y-axis scale.
    """
    if not episode_stats:
        print("No stats to plot.")
        return

    # Extract data
    episodes = [e['episode_number'] for e in episode_stats]
    rewards = [e.get('reward', np.nan) for e in episode_stats]
    
    # Calculate rolling average reward (100 episodes)
    avg_rewards = []
    if len(rewards) > 0:
        avg_rewards = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
    
    # Check if additional metrics are available
    first_valid_stat_entry = next((item for item in episode_stats if isinstance(item, dict)), None)
    has_avg_max_q = 'avg_max_q' in first_valid_stat_entry if first_valid_stat_entry else False
    has_avg_loss = 'avg_loss' in first_valid_stat_entry if first_valid_stat_entry else False

    # Create figure with primary y-axis
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.suptitle('Pong DQN Training Progress', fontsize=16)

    # Plot rewards on primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(episodes, rewards, label='Episode Reward', alpha=0.6, color=color, linewidth=1)
    if avg_rewards:
        ax1.plot(episodes, avg_rewards, label='Avg Reward (100ep)', color='red', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Prepare for legend
    lines, labels = ax1.get_legend_handles_labels()
    
    # Initialize secondary y-axes
    ax2 = None
    ax3 = None 

    # Plot avg_max_q if available
    if has_avg_max_q:
        avg_max_q_values = [e.get('avg_max_q', np.nan) for e in episode_stats]
        ax2 = ax1.twinx() 
        color = 'tab:green'
        ax2.set_ylabel('Avg Max Q', color=color)
        line2_plot, = ax2.plot(episodes, avg_max_q_values, label='Avg Max Q (ep)', 
                              color=color, linestyle='--', linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor=color)
        lines.append(line2_plot)
        labels.append('Avg Max Q (ep)')

    # Plot avg_loss if available
    if has_avg_loss:
        avg_loss_values = [e.get('avg_loss', np.nan) for e in episode_stats]
        ax3 = ax1.twinx() 
        
        # Offset the right spine of ax3 if ax2 exists
        if ax2 is not None: 
            ax3.spines["right"].set_position(("outward", 60)) 
        
        color = 'tab:purple'
        ax3.set_ylabel('Avg Loss', color=color)
        line3_plot, = ax3.plot(episodes, avg_loss_values, label='Avg Loss (ep)', 
                              color=color, linestyle=':', linewidth=1.5)
        ax3.tick_params(axis='y', labelcolor=color)
        lines.append(line3_plot)
        labels.append('Avg Loss (ep)')

    # Add legend
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              fancybox=True, shadow=True, ncol=min(3, len(lines)))
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0.1, 1, 0.95]) 
    
    # Save and close
    plt.savefig(filename)
    plt.close(fig) 
    print(f"Plot saved to {filename}")
