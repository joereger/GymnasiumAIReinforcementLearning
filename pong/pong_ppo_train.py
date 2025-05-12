import gymnasium as gym
import torch
import numpy as np
import random
import time
import os
import json

# Import from other modules
from pong_ppo_utils import make_atari_env, RolloutBuffer, format_time
from pong_ppo_model import PPOAgent, device
from pong_ppo_vis import plot_training_data, save_training_stats, load_training_stats

# --- Evaluation Function ---
def evaluate_ppo_agent(env_name, agent, num_episodes=10, human_render=False):
    """
    Evaluate a trained PPO agent on the Pong environment.
    
    Args:
        env_name: Name of the environment
        agent: PPO agent instance
        num_episodes: Number of episodes to evaluate
        human_render: Whether to render the environment for human viewing
    
    Returns:
        The average reward over the evaluation episodes
    """
    print(f"\nEvaluating PPO agent for {num_episodes} episodes...")
    eval_render_mode = "human" if human_render else "rgb_array"
    
    # Create properly wrapped evaluation environment
    eval_env = make_atari_env(env_name, render_mode=eval_render_mode)
    
    # Run evaluation episodes
    total_rewards = []
    
    for episode in range(num_episodes):
        state, info = eval_env.reset(seed=random.randint(0, 1_000_000))
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Select greedy action (deterministic)
            action, _, _ = agent.act(state, deterministic=True)
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if human_render:
                eval_env.render()
                
            if steps > 20000:
                print("Warning: Evaluation episode exceeded 20000 steps.")
                break
                
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    # Clean up
    eval_env.close()
    avg_reward = np.mean(total_rewards)
    print(f"Average evaluation reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

# --- Training Function ---
def train_ppo_agent(
    rollout_length=128,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_param=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    ppo_epochs=4,
    batch_size=32,
    max_episodes=5000,
    eval_interval=10,
    save_interval=10,
    human_render_during_training=False,
    load_checkpoint_flag=False
):
    """
    Train a PPO agent on the Pong environment.
    
    Args:
        rollout_length: Number of steps to collect for each PPO update
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_param: PPO clipping parameter
        value_coef: Value function coefficient in loss
        entropy_coef: Entropy bonus coefficient
        max_grad_norm: Maximum gradient norm for clipping
        ppo_epochs: Number of optimization epochs per update
        batch_size: Minibatch size for PPO updates
        max_episodes: Maximum number of episodes to train for
        eval_interval: Evaluate every N episodes
        save_interval: Save checkpoint every N episodes
        human_render_during_training: Whether to render during training (slower)
        load_checkpoint_flag: Whether to load from checkpoint
    """
    # Environment and hyperparameters
    ENV_NAME = "PongNoFrameskip-v4"
    train_render_mode = "human" if human_render_during_training else None
    
    # Seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    elif device.type == 'mps':
        # MPS doesn't support seed_all, individual tensors will be seeded
        pass
    
    # Setup paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pong")
    os.makedirs(DATA_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(DATA_DIR, "pong_ppo_model.pth")
    BEST_MODEL_PATH = os.path.join(DATA_DIR, "pong_ppo_best_model.pth")
    STATS_PATH = os.path.join(DATA_DIR, "pong_ppo_stats.json")
    PLOT_PATH = os.path.join(DATA_DIR, "pong_ppo_progress.png")
    
    # Create properly wrapped environment
    env = make_atari_env(ENV_NAME, render_mode=train_render_mode)
    ACTION_SPACE_SIZE = env.action_space.n
    print(f"Action space size: {ACTION_SPACE_SIZE}")
    
    # State shape is (4, 84, 84) for channels-first PyTorch input
    STATE_SHAPE = (4, 84, 84)
    
    # Create PPO agent
    agent = PPOAgent(
        state_shape=STATE_SHAPE,
        action_space=ACTION_SPACE_SIZE,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_param=clip_param,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size
    )
    
    # Initialize training stats
    stats = {
        "episode_rewards": [],
        "episode_lengths": [],
        "evaluation_rewards": [],
        "entropies": [],
        "policy_losses": [],
        "value_losses": [],
        "total_losses": [],
        "update_timestamps": [],
        "hyperparams": {
            "rollout_length": rollout_length,
            "lr": lr,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_param": clip_param,
            "value_coef": value_coef,
            "entropy_coef": entropy_coef,
            "ppo_epochs": ppo_epochs,
            "batch_size": batch_size
        },
        "total_steps": 0,
        "best_eval_reward": -float('inf'),
        "training_start": time.time(),
        "cumulative_training_time": 0.0,
        "experiment_version": "PPO - Experiment 3"
    }
    
    start_episode = 1
    best_eval_reward = -float('inf')
    total_steps = 0
    
    # Load checkpoint if requested
    if load_checkpoint_flag:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model checkpoint from {MODEL_PATH}")
            agent.load(MODEL_PATH)
        else:
            print(f"Checkpoint {MODEL_PATH} not found. Starting new training.")
        
        if os.path.exists(STATS_PATH):
            print(f"Loading training stats from {STATS_PATH}")
            loaded_stats = load_training_stats(STATS_PATH)
            
            if loaded_stats:
                # Copy over the stats we want to continue tracking
                stats["episode_rewards"] = loaded_stats.get("episode_rewards", [])
                stats["episode_lengths"] = loaded_stats.get("episode_lengths", [])
                stats["evaluation_rewards"] = loaded_stats.get("evaluation_rewards", [])
                stats["entropies"] = loaded_stats.get("entropies", [])
                stats["policy_losses"] = loaded_stats.get("policy_losses", [])
                stats["value_losses"] = loaded_stats.get("value_losses", [])
                stats["total_losses"] = loaded_stats.get("total_losses", [])
                stats["update_timestamps"] = loaded_stats.get("update_timestamps", [])
                stats["cumulative_training_time"] = loaded_stats.get("cumulative_training_time", 0.0)
                stats["total_steps"] = loaded_stats.get("total_steps", 0)
                
                # Set variables for resuming
                start_episode = len(stats["episode_rewards"]) + 1
                best_eval_reward = loaded_stats.get("best_eval_reward", -float('inf'))
                total_steps = stats["total_steps"]
                
                print(f"Resuming from episode {start_episode}, total steps: {total_steps}, best eval: {best_eval_reward}")
    
    # Create rollout buffer for collecting experiences
    rollout_buffer = RolloutBuffer(rollout_length, STATE_SHAPE, ACTION_SPACE_SIZE, device)
    
    # Start training timer
    training_start_time = time.time()
    last_update_time = training_start_time
    
    print(f"\nStarting PPO training from episode {start_episode} for up to {max_episodes} episodes...")
    print(f"PPO Hyperparameters:")
    print(f"  Rollout Length: {rollout_length}")
    print(f"  Learning Rate: {lr}")
    print(f"  Gamma: {gamma}")
    print(f"  GAE Lambda: {gae_lambda}")
    print(f"  Clip Parameter: {clip_param}")
    print(f"  Value Coefficient: {value_coef}")
    print(f"  Entropy Coefficient: {entropy_coef}")
    print(f"  PPO Epochs: {ppo_epochs}")
    print(f"  Batch Size: {batch_size}")
    
    # Main training loop
    current_episode = start_episode
    
    while current_episode <= max_episodes:
        # Reset environment for new episode
        state, info = env.reset(seed=SEED + current_episode)
        
        # Initialize episode stats
        episode_reward = 0
        episode_length = 0
        episode_start_time = time.time()
        
        # Episode loop
        done = False
        
        while not done:
            # Initialize temporary lists for collecting rollout data
            batch_states = []
            batch_actions = []
            batch_log_probs = []
            batch_rewards = []
            batch_dones = []
            batch_values = []
            
            # Collect rollout of fixed length
            for step in range(rollout_length):
                # Get action from agent
                action, log_prob, value = agent.act(state)
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition in rollout buffer
                rollout_buffer.store(state, action, reward, float(done), log_prob, value)
                
                # Update state
                state = next_state
                
                # Update episode stats
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # If episode is done, break rollout collection
                if done:
                    break
                    
                # Render if requested
                if human_render_during_training:
                    env.render()
            
            # If we completed a rollout, calculate advantages and returns
            if rollout_buffer.idx > 0:  # Only if we have some data
                # Get last value for bootstrapping (unless episode ended)
                if not done:
                    _, _, last_value = agent.act(state)
                else:
                    last_value = 0.0
                
                # Calculate advantages and returns using GAE
                rollout_buffer.compute_advantages_and_returns(last_value, gamma, gae_lambda)
                
                # Get batch of data for training
                batch_data = rollout_buffer.get_batch()
                
                if batch_data is not None:
                    # Update policy using PPO
                    loss_stats = agent.update(batch_data)
                    
                    # Record loss stats
                    stats["policy_losses"].append(loss_stats["policy_loss"])
                    stats["value_losses"].append(loss_stats["value_loss"])
                    stats["entropies"].append(loss_stats["entropy"])
                    stats["total_losses"].append(loss_stats["total_loss"])
                    stats["update_timestamps"].append(time.time())
                
                # Clear rollout buffer after update
                rollout_buffer.clear()
            
            # Break if episode ended
            if done:
                break
        
        # Record episode stats
        episode_duration = time.time() - episode_start_time
        stats["episode_rewards"].append(episode_reward)
        stats["episode_lengths"].append(episode_length)
        stats["total_steps"] = total_steps
        
        # Calculate cumulative training time
        current_training_time = time.time() - training_start_time
        stats["cumulative_training_time"] = stats.get("cumulative_training_time", 0.0) + current_training_time
        
        # Calculate average reward over last 100 episodes
        last_100_rewards = stats["episode_rewards"][-100:]
        avg_reward = np.mean(last_100_rewards)
        
        # Print episode stats
        print(f"Episode: {current_episode}/{max_episodes} | "
              f"Steps: {episode_length} | "
              f"Total Steps: {total_steps} | "
              f"Reward: {episode_reward:.2f} | "
              f"Avg Reward (100ep): {avg_reward:.2f} | "
              f"Entropy: {stats['entropies'][-1]:.6f} | "
              f"Policy Loss: {stats['policy_losses'][-1]:.6f} | "
              f"Value Loss: {stats['value_losses'][-1]:.6f} | "
              f"Time: {format_time(stats['cumulative_training_time'])}")
        
        # Periodic evaluation
        if current_episode % eval_interval == 0:
            eval_reward = evaluate_ppo_agent(ENV_NAME, agent, num_episodes=5, human_render=False)
            stats["evaluation_rewards"].append(eval_reward)
            stats["best_eval_reward"] = max(stats["best_eval_reward"], eval_reward)
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                # Save best model
                agent.save(BEST_MODEL_PATH)
                print(f"New best evaluation reward: {best_eval_reward:.2f}. Model saved to {BEST_MODEL_PATH}")
            
            # Update plot
            plot_training_data(stats, PLOT_PATH)
        
        # Periodic saving
        if current_episode % save_interval == 0:
            # Save checkpoint
            agent.save(MODEL_PATH)
            
            # Save stats
            save_training_stats(stats, STATS_PATH)
            print(f"Checkpoint saved at episode {current_episode}")
        
        # Increment episode counter
        current_episode += 1
    
    # Final cleanup and saving
    env.close()
    
    # Save final model and stats
    agent.save(MODEL_PATH)
    save_training_stats(stats, STATS_PATH)
    
    # Create final plot
    plot_training_data(stats, PLOT_PATH)
    
    print(f"Training completed after {format_time(stats['cumulative_training_time'])}")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Total steps: {total_steps}")
    
    return agent, stats
