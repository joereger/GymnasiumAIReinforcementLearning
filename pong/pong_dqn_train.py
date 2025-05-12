import gymnasium as gym
import torch
import numpy as np
import random
import time
import os
import json
import ale_py

# Import from other modules
from pong_dqn_utils import make_atari_env, ReplayBuffer, format_time
from pong_dqn_model import DQNAgent, device
from pong_dqn_vis import plot_training_data

# --- Evaluation Function ---
def evaluate_pong_agent(env_name, model_path_to_load, num_episodes=10, human_render=False, eval_epsilon=0.05):
    """
    Evaluate a trained DQN agent on the Pong environment.
    
    Args:
        env_name: Name of the environment
        model_path_to_load: Path to the saved model weights
        num_episodes: Number of episodes to evaluate
        human_render: Whether to render the environment for human viewing
        eval_epsilon: Small exploration rate during evaluation (e.g., 0.05)
                      Set to 0 for pure greedy policy
    
    Returns:
        The average reward over the evaluation episodes
    """
    print(f"\nEvaluating agent with model: {model_path_to_load} for {num_episodes} episodes...")
    print(f"Using evaluation epsilon: {eval_epsilon}")
    eval_render_mode = "human" if human_render else "rgb_array"
    
    # Register ALE environments if needed
    if 'ale_py' in globals() and hasattr(ale_py, '__version__'):
        gym.register_envs(ale_py)
        
    # Create properly wrapped evaluation environment with reduced actions
    eval_env = make_atari_env(env_name, render_mode=eval_render_mode, reduced_actions=True)
    
    # Get env properties
    ACTION_SPACE_SIZE = eval_env.action_space.n
    print(f"Evaluation environment action space size: {ACTION_SPACE_SIZE}")
    
    # Create agent with the proper state shape (4, 84, 84)
    STATE_SHAPE_EVAL = (4, 84, 84)  # Channels-first shape for PyTorch

    agent_eval = DQNAgent(
        state_shape=STATE_SHAPE_EVAL,
        action_space=ACTION_SPACE_SIZE 
    )
    
    # Load model weights
    if os.path.exists(model_path_to_load):
        print(f"Loading model for evaluation from {model_path_to_load}")
        agent_eval.load(model_path_to_load)
    else:
        print(f"ERROR: Model not found at {model_path_to_load}. Cannot evaluate.")
        eval_env.close()
        return -float('inf')

    # Set to evaluation mode
    agent_eval.policy_net.eval() 

    # Run evaluation episodes
    total_rewards = []
    action_distribution = {}
    
    for episode in range(num_episodes):
        state, info = eval_env.reset(seed=random.randint(0, 1_000_000)) 
        done = False
        episode_reward = 0
        steps = 0
        episode_actions = []
        
        while not done:
            # Allow a small amount of exploration during evaluation
            if random.random() < eval_epsilon:
                action = random.randrange(ACTION_SPACE_SIZE)
                exploration_type = "random"
            else:
                action = agent_eval.act(state, explore=False)
                exploration_type = "greedy"
            
            episode_actions.append(action)
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
        
        # Count actions taken
        episode_action_counts = {}
        for a in range(ACTION_SPACE_SIZE):
            count = episode_actions.count(a)
            episode_action_counts[a] = count
            action_distribution[a] = action_distribution.get(a, 0) + count
                
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}: " +
              f"Reward = {episode_reward:.2f}, Steps = {steps}, " +
              f"Actions = {episode_action_counts}")

    # Clean up and report results
    eval_env.close()
    avg_reward = np.mean(total_rewards)
    print(f"Average evaluation reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Overall action distribution: {action_distribution}")
    return avg_reward

# --- Training Function ---
def train_pong_agent(human_render_during_training=False, load_checkpoint_flag=False):
    """
    Train a DQN agent on the Pong environment.
    
    Args:
        human_render_during_training: Whether to render during training
        load_checkpoint_flag: Whether to load a saved checkpoint
        
    Returns:
        The trained agent
    """
    # Environment and hyperparameters
    ENV_NAME = "PongNoFrameskip-v4"
    train_render_mode = "human" if human_render_during_training else None
    LEARNING_RATE = 2.5e-4
    BATCH_SIZE = 32
    REPLAY_BUFFER_SIZE = int(5e4)  # Reduced
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 1000     # Reduced
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_FRAMES = int(1e5) # Reduced
    MAX_EPISODES = 5000
    MAX_FRAMES_TOTAL = int(5e5)   # Reduced
    EVAL_INTERVAL_EPISODES = 10 
    SAVE_INTERVAL_EPISODES = 10 
    FRAMES_PER_STATE = 4

    # Paths for models and data
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pong")
    os.makedirs(DATA_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(DATA_DIR, "pong_dqn_model.pth") 
    BEST_MODEL_PATH = os.path.join(DATA_DIR, "pong_dqn_best_model.pth")
    REWARDS_PLOT_PATH = os.path.join(DATA_DIR, "pong_training_progress.png") 
    STATS_PATH = os.path.join(DATA_DIR, "pong_training_stats.json") 

    # Set seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type != 'cpu':
        torch.cuda.manual_seed_all(SEED)

    # Register ALE environments if needed
    if 'ale_py' in globals() and hasattr(ale_py, '__version__'):
        gym.register_envs(ale_py)
    
    # Create properly wrapped environment with reduced action space
    env = make_atari_env(ENV_NAME, render_mode=train_render_mode, reduced_actions=True)
    ACTION_SPACE_SIZE = env.action_space.n
    print(f"Training with reduced action space: {ACTION_SPACE_SIZE} actions")

    # State shape is (4, 84, 84) for channels-first PyTorch input
    STATE_SHAPE = (4, 84, 84)

    # Create agent
    agent = DQNAgent(
        state_shape=STATE_SHAPE,
        action_space=ACTION_SPACE_SIZE,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        target_update_freq=TARGET_UPDATE_FREQ, 
        epsilon_start=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay_frames=EPSILON_DECAY_FRAMES,
        batch_size=BATCH_SIZE
    )
    
    # Initialize training stats
    all_episode_stats = [] 
    start_episode = 1
    cumulative_training_time_seconds_loaded = 0.0
    best_eval_reward = -float('inf') 

    # Load checkpoint if requested
    if load_checkpoint_flag:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model checkpoint from {MODEL_PATH}")
            agent.load(MODEL_PATH) 
        else:
            print(f"Checkpoint {MODEL_PATH} not found. Starting new training.")
        
        if os.path.exists(STATS_PATH):
            print(f"Loading training stats from {STATS_PATH}")
            try:
                with open(STATS_PATH, 'r') as f:
                    stats = json.load(f)
                all_episode_stats = stats.get("episode_stats", [])
                agent.current_frames = stats.get("total_agent_steps_completed", 0) 
                start_episode = stats.get("last_completed_episode_number", 0) + 1
                best_eval_reward = stats.get("best_eval_reward_achieved", -float('inf'))
                cumulative_training_time_seconds_loaded = stats.get("cumulative_training_time_seconds", 0.0)
                print(f"Resuming from episode {start_episode}, total steps: {agent.current_frames}, best_eval_reward: {best_eval_reward:.2f}, cumulative_time: {format_time(cumulative_training_time_seconds_loaded)}")
            except json.JSONDecodeError:
                print(f"Error decoding {STATS_PATH}. Starting with fresh stats.")
        else:
            print(f"Stats file {STATS_PATH} not found. Starting with fresh stats.")

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(size=REPLAY_BUFFER_SIZE)
    
    # No warmup phase, learning starts once buffer has BATCH_SIZE samples

    # Start training timer
    session_start_time = time.time()

    print(f"Starting training from episode {start_episode} up to {MAX_EPISODES} or until {MAX_FRAMES_TOTAL} agent steps...")
    if human_render_during_training:
        print("Human rendering enabled during training. This will be slower.")

    # Main training loop
    for episode_num in range(start_episode, MAX_EPISODES + 1):
        episode_start_time = time.time()
        state, info = env.reset(seed=SEED + episode_num)
        done = False
        current_episode_reward = 0
        current_episode_steps = 0
        episode_losses = []
        episode_max_q_values = []

        # Episode loop
        while not done:
            # Get max Q-value for current state (for logging)
            with torch.no_grad():
                state_tensor_for_q = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values_pred = agent.policy_net(state_tensor_for_q)
                episode_max_q_values.append(q_values_pred.max().item())

            # Select and perform action
            action = agent.act(state) 
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if human_render_during_training:
                env.render()

            # Store transition - reward is already clipped by the ClipRewardEnv wrapper
            replay_buffer.store((state, action, reward, next_state, float(done)))

            # Update state
            state = next_state
            current_episode_reward += reward
            current_episode_steps += 1
            
            # Update network
            loss = agent.learn(replay_buffer)
            if loss is not None:
                episode_losses.append(loss)

            # Check if we've reached max frames
            if agent.current_frames >= MAX_FRAMES_TOTAL:
                print(f"Reached max total agent steps ({MAX_FRAMES_TOTAL}). Stopping training.")
                done = True
        
        # Episode completion stats
        episode_duration_seconds = time.time() - episode_start_time
        current_session_active_time = time.time() - session_start_time
        total_cumulative_time_seconds = cumulative_training_time_seconds_loaded + current_session_active_time
        
        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0
        avg_episode_max_q = np.mean(episode_max_q_values) if episode_max_q_values else 0
        
        # Record episode data
        episode_data = {
            "episode_number": episode_num,
            "reward": current_episode_reward,
            "steps": current_episode_steps,
            "epsilon": agent.get_epsilon(),
            "timestamp_end_episode": time.time(),
            "duration_episode_seconds": episode_duration_seconds,
            "avg_loss": avg_episode_loss,
            "avg_max_q": avg_episode_max_q
        }
        all_episode_stats.append(episode_data)
        
        # Calculate average reward for logging
        rewards_for_plot = [e['reward'] for e in all_episode_stats]
        avg_reward_for_plot = np.mean(rewards_for_plot[-100:]) if len(rewards_for_plot) > 0 else np.nan
        
        # Print episode stats
        print(f"Episode: {episode_num}/{MAX_EPISODES} | "
              f"Steps: {current_episode_steps} | "
              f"Total Steps: {agent.current_frames} | "
              f"Reward: {current_episode_reward:.2f} | "
              f"Avg Reward (100ep): {avg_reward_for_plot:.2f} | "
              f"Epsilon: {agent.get_epsilon():.4f} | "
              f"Avg Loss: {avg_episode_loss:.4f} | "
              f"Avg Max Q: {avg_episode_max_q:.2f} | "
              f"Cum. Time: {format_time(total_cumulative_time_seconds)}")

        # Periodic evaluation
        if episode_num % EVAL_INTERVAL_EPISODES == 0:
            current_model_to_eval_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
            if not os.path.exists(current_model_to_eval_path) and episode_num >= SAVE_INTERVAL_EPISODES:
                 agent.save(MODEL_PATH) 
                 current_model_to_eval_path = MODEL_PATH
            
            if os.path.exists(current_model_to_eval_path):
                 eval_reward_val = evaluate_pong_agent(ENV_NAME, current_model_to_eval_path, num_episodes=5, human_render=False)
                 print(f"Evaluation after Episode {episode_num}: Avg Reward = {eval_reward_val:.2f}")
                 if eval_reward_val > best_eval_reward:
                      best_eval_reward = eval_reward_val
                      agent.save(BEST_MODEL_PATH) 
                      print(f"New best evaluation reward: {best_eval_reward:.2f}. Best model saved to {BEST_MODEL_PATH}")
            else:
                 print(f"Skipping evaluation at episode {episode_num} as no model saved yet ({current_model_to_eval_path}).")
            
            # Update plot
            plot_training_data(all_episode_stats, filename=REWARDS_PLOT_PATH)

        # Periodic saving
        if episode_num % SAVE_INTERVAL_EPISODES == 0:
            agent.save(MODEL_PATH)
            print(f"Model checkpoint saved at episode {episode_num} to {MODEL_PATH}")
            stats_to_save = {
                "episode_stats": all_episode_stats,
                "total_agent_steps_completed": agent.current_frames,
                "last_completed_episode_number": episode_num,
                "best_eval_reward_achieved": best_eval_reward,
                "cumulative_training_time_seconds": total_cumulative_time_seconds
            }
            with open(STATS_PATH, 'w') as f:
                json.dump(stats_to_save, f, indent=4)
            print(f"Training stats saved to {STATS_PATH}")

        # Check if we've reached max frames globally
        if agent.current_frames >= MAX_FRAMES_TOTAL:
            break

    # Cleanup and final save
    env.close()
    print("Training finished.")
    agent.save(MODEL_PATH)
    final_total_cumulative_time = cumulative_training_time_seconds_loaded + (time.time() - session_start_time)
    final_stats_to_save = {
        "episode_stats": all_episode_stats,
        "total_agent_steps_completed": agent.current_frames,
        "last_completed_episode_number": MAX_EPISODES if agent.current_frames < MAX_FRAMES_TOTAL else episode_num, 
        "best_eval_reward_achieved": best_eval_reward,
        "cumulative_training_time_seconds": final_total_cumulative_time
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(final_stats_to_save, f, indent=4)
    print(f"Final model saved to {MODEL_PATH}")
    print(f"Final training stats saved to {STATS_PATH}")
    
    # Final plot
    plot_training_data(all_episode_stats, filename=REWARDS_PLOT_PATH)
    print(f"Final plot saved to {REWARDS_PLOT_PATH}")
    return agent
