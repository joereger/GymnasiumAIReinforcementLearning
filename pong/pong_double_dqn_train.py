import gymnasium as gym
import torch
import numpy as np
import random
import time
import os
import json
import ale_py

# Import from other modules
from pong_double_dqn_utils import FrameStack, ReplayBuffer, format_time, preprocess
from pong_double_dqn_model import DoubleDQNAgent, device
from pong_double_dqn_vis import plot_training_data

# --- Evaluation Function ---
def evaluate_double_dqn_agent(env_name, model_path_to_load, num_episodes=10, human_render=False):
    """
    Evaluate a trained Double DQN agent on the Pong environment.
    
    Args:
        env_name: Name of the environment
        model_path_to_load: Path to the saved model weights
        num_episodes: Number of episodes to evaluate
        human_render: Whether to render the environment for human viewing
    
    Returns:
        The average reward over the evaluation episodes
    """
    print(f"\nEvaluating Double DQN agent with model: {model_path_to_load} for {num_episodes} episodes...")
    eval_render_mode = "human" if human_render else "rgb_array"
    
    # Register ALE environments if needed
    if 'ale_py' in globals() and hasattr(ale_py, '__version__'):
        gym.register_envs(ale_py)
        
    # Create evaluation environment
    eval_env = gym.make(env_name, render_mode=eval_render_mode, repeat_action_probability=0.0)
    
    # Get env properties
    ACTION_SPACE_SIZE = eval_env.action_space.n
    frame_stacker_eval = FrameStack(k=4) 
    
    # Create agent with same state shape as training
    temp_obs_eval, _ = eval_env.reset()
    temp_state_eval = frame_stacker_eval.reset(temp_obs_eval)
    STATE_SHAPE_EVAL = temp_state_eval.shape

    agent_eval = DoubleDQNAgent(
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
    for episode in range(num_episodes):
        obs, info = eval_env.reset(seed=random.randint(0, 1_000_000)) 
        state = frame_stacker_eval.reset(obs)
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Select greedy action (no exploration)
            action = agent_eval.act(state, explore=False) 
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            state = frame_stacker_eval.step(next_obs)
            episode_reward += reward
            steps += 1
            
            if human_render: 
                eval_env.render() 
                
            if steps > 20000: 
                print("Warning: Evaluation episode exceeded 20000 steps.")
                break
                
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}")

    # Clean up and return results
    eval_env.close()
    avg_reward = np.mean(total_rewards)
    print(f"Average evaluation reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

# --- Training Function ---
def train_double_dqn_agent(human_render_during_training=False, load_checkpoint_flag=False):
    """
    Train a Double DQN agent on the Pong environment.
    
    Key differences from regular DQN (Experiment 2):
    - Uses Double DQN algorithm to reduce Q-value overestimation bias
    - Lower learning rate (1e-5)
    - More frequent target network updates (1,000 steps)
    - Larger replay buffer (500K)
    - Slower epsilon decay (2M frames)
    
    Args:
        human_render_during_training: Whether to render during training
        load_checkpoint_flag: Whether to load a saved checkpoint
    """
    # Environment and hyperparameters
    ENV_NAME = "PongNoFrameskip-v4"
    train_render_mode = "human" if human_render_during_training else None
    LEARNING_RATE = 1e-5  # Reduced from 2.5e-4 for Experiment 2
    BATCH_SIZE = 32
    REPLAY_BUFFER_SIZE = int(5e5)  # Increased from 1e5 for Experiment 2
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 1000  # Decreased from 10,000 for Experiment 2
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_FRAMES = int(2e6)  # Increased from 1e6 for Experiment 2
    MAX_EPISODES = 5000
    MAX_FRAMES_TOTAL = int(2e6)
    EVAL_INTERVAL_EPISODES = 10 
    SAVE_INTERVAL_EPISODES = 10 
    FRAMES_PER_STATE = 4

    # Paths for models and data
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pong")
    os.makedirs(DATA_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(DATA_DIR, "pong_double_dqn_model.pth") 
    BEST_MODEL_PATH = os.path.join(DATA_DIR, "pong_double_dqn_best_model.pth")
    REWARDS_PLOT_PATH = os.path.join(DATA_DIR, "pong_double_dqn_progress.png") 
    STATS_PATH = os.path.join(DATA_DIR, "pong_double_dqn_stats.json") 

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
    
    # Create environment
    env = gym.make(ENV_NAME, render_mode=train_render_mode, repeat_action_probability=0.0)
    ACTION_SPACE_SIZE = env.action_space.n
    print(f"Action space size: {ACTION_SPACE_SIZE}")

    # Initialize frame stacker
    frame_stacker = FrameStack(k=FRAMES_PER_STATE) 
    temp_obs, _ = env.reset(seed=SEED)
    temp_state = frame_stacker.reset(temp_obs)
    STATE_SHAPE = temp_state.shape

    # Create Double DQN agent
    agent = DoubleDQNAgent(
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
    
    # Warm up replay buffer if needed
    WARMUP_STEPS = 50000 
    if not load_checkpoint_flag or agent.current_frames < WARMUP_STEPS : 
        print(f"Warming up replay buffer with random policy for {WARMUP_STEPS} steps...")
        warmup_seed = SEED - 1 if not load_checkpoint_flag else SEED + start_episode + agent.current_frames
        obs_warmup, _ = env.reset(seed=warmup_seed) 
        state_warmup = frame_stacker.reset(obs_warmup)
        frames_done_warmup = 0
        while frames_done_warmup < WARMUP_STEPS:
            action_warmup = env.action_space.sample()
            next_obs_warmup, reward_warmup, terminated_warmup, truncated_warmup, _ = env.step(action_warmup)
            done_warmup = terminated_warmup or truncated_warmup
            
            next_state_warmup = frame_stacker.step(next_obs_warmup)
            clipped_reward_warmup = np.sign(reward_warmup) 
            replay_buffer.store((state_warmup, action_warmup, clipped_reward_warmup, next_state_warmup, float(done_warmup)))
            
            if done_warmup:
                warmup_seed += 1 
                obs_warmup, _ = env.reset(seed=warmup_seed) 
                state_warmup = frame_stacker.reset(obs_warmup)
            else:
                state_warmup = next_state_warmup
            frames_done_warmup +=1
            if frames_done_warmup % 1000 == 0:
                print(f"Warmup step {frames_done_warmup}/{WARMUP_STEPS}")
        print("Replay buffer warmup complete.")

    # Start training timer
    session_start_time = time.time()

    print(f"Starting Double DQN training from episode {start_episode} up to {MAX_EPISODES} or until {MAX_FRAMES_TOTAL} agent steps...")
    print(f"Using: LR={LEARNING_RATE}, Target Update={TARGET_UPDATE_FREQ}, Epsilon Decay={EPSILON_DECAY_FRAMES/1e6}M frames")
    if human_render_during_training:
        print("Human rendering enabled during training. This will be slower.")

    # Main training loop
    for episode_num in range(start_episode, MAX_EPISODES + 1):
        episode_start_time = time.time()
        obs, info = env.reset(seed=SEED + episode_num)
        state = frame_stacker.reset(obs)
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
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if human_render_during_training:
                env.render()

            # Process transition
            next_state = frame_stacker.step(next_obs)
            clipped_reward = np.sign(reward) 
            replay_buffer.store((state, action, clipped_reward, next_state, float(done)))

            # Update state
            state = next_state
            current_episode_reward += reward
            current_episode_steps += 1
            
            # Update network with Double DQN learning
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
              f"Epsilon: {agent.get_epsilon():.6f} | "
              f"Avg Loss: {avg_episode_loss:.6f} | "
              f"Avg Max Q: {avg_episode_max_q:.6f} | "
              f"Cum. Time: {format_time(total_cumulative_time_seconds)}")

        # Periodic evaluation
        if episode_num % EVAL_INTERVAL_EPISODES == 0:
            current_model_to_eval_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
            if not os.path.exists(current_model_to_eval_path) and episode_num >= SAVE_INTERVAL_EPISODES:
                 agent.save(MODEL_PATH) 
                 current_model_to_eval_path = MODEL_PATH
            
            if os.path.exists(current_model_to_eval_path):
                 eval_reward_val = evaluate_double_dqn_agent(ENV_NAME, current_model_to_eval_path, num_episodes=5, human_render=False)
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
                "cumulative_training_time_seconds": total_cumulative_time_seconds,
                "experiment_version": "Double DQN - Experiment 2",
                "hyperparams": {
                    "learning_rate": LEARNING_RATE,
                    "target_update_freq": TARGET_UPDATE_FREQ,
                    "epsilon_decay_frames": EPSILON_DECAY_FRAMES,
                    "replay_buffer_size": REPLAY_BUFFER_SIZE
                }
            }
            with open(STATS_PATH, 'w') as f:
                json.dump(stats_to_save, f, indent=4)
            print(f"Training stats saved to {STATS_PATH}")

        # Check if we've reached max frames globally
        if agent.current_frames >= MAX_FRAMES_TOTAL:
            break

    # Cleanup and final save
    env.close()
    print("Double DQN training finished.")
    agent.save(MODEL_PATH)
    final_total_cumulative_time = cumulative_training_time_seconds_loaded + (time.time() - session_start_time)
    final_stats_to_save = {
        "episode_stats": all_episode_stats,
        "total_agent_steps_completed": agent.current_frames,
        "last_completed_episode_number": MAX_EPISODES if agent.current_frames < MAX_FRAMES_TOTAL else episode_num, 
        "best_eval_reward_achieved": best_eval_reward,
        "cumulative_training_time_seconds": final_total_cumulative_time,
        "experiment_version": "Double DQN - Experiment 2",
        "hyperparams": {
            "learning_rate": LEARNING_RATE, 
            "target_update_freq": TARGET_UPDATE_FREQ,
            "epsilon_decay_frames": EPSILON_DECAY_FRAMES,
            "replay_buffer_size": REPLAY_BUFFER_SIZE
        }
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(final_stats_to_save, f, indent=4)
    print(f"Final model saved to {MODEL_PATH}")
    print(f"Final training stats saved to {STATS_PATH}")
    
    # Final plot
    plot_training_data(all_episode_stats, filename=REWARDS_PLOT_PATH)
    print(f"Final plot saved to {REWARDS_PLOT_PATH}")
    return agent
