import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import random
from collections import deque
import time
import matplotlib.pyplot as plt
import os
import json # For saving/loading stats
import ale_py # Import ale_py

# --- Device Configuration ---
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# --- Preprocessing ---
def preprocess(frame):
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame

# --- Frame Stacking ---
class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        processed_obs = preprocess(obs)
        for _ in range(self.k):
            self.frames.append(processed_obs)
        return np.stack(list(self.frames), axis=0)

    def step(self, obs):
        self.frames.append(preprocess(obs))
        return np.stack(list(self.frames), axis=0)

# --- DQN Model ---
class PongDQN(nn.Module):
    def __init__(self, input_channels=4, action_space=6): # Default action_space, will be updated
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.fc_input_dims = 7 * 7 * 64
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512), nn.ReLU(inplace=True),
            nn.Linear(512, action_space),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, size=int(1e5)):
        self.buffer = deque(maxlen=size)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_shape, action_space, lr=2.5e-4, gamma=0.99, target_update_freq=1000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_frames=1e6, batch_size=32):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.policy_net = PongDQN(input_channels=state_shape[0], action_space=action_space).to(device)
        self.target_net = PongDQN(input_channels=state_shape[0], action_space=action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        self.current_frames = 0

    def get_epsilon(self):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.current_frames / self.epsilon_decay_frames)
        return epsilon

    def act(self, state, explore=True):
        self.current_frames += 1
        epsilon = self.get_epsilon() if explore else 0.0
        
        if random.random() < epsilon:
            return random.randrange(self.action_space)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def learn(self, replay_buffer):
        if len(replay_buffer) < self.batch_size:
            return None

        transitions = replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions)) # state, action, reward, next_state, done

        state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0) # Gradient clipping enabled
        self.optimizer.step()

        if self.current_frames % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Ensure target net is also updated

# --- Plotting ---
def plot_rewards(rewards, avg_rewards, filename="pong_rewards.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Avg Reward (Last 100 episodes)', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Pong DQN Training Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved to {filename}")

# --- Helper to format time ---
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# --- Evaluation ---
def evaluate_pong_agent(env_name, model_path_to_load, num_episodes=10, human_render=False):
    print(f"\nEvaluating agent with model: {model_path_to_load} for {num_episodes} episodes...")
    eval_render_mode = "human" if human_render else "rgb_array"
    
    # Ensure ALE is registered for this new env instance
    if 'ale_py' in globals() and hasattr(ale_py, '__version__'):
        gym.register_envs(ale_py)
        
    eval_env = gym.make(env_name, render_mode=eval_render_mode, repeat_action_probability=0.0)
    
    ACTION_SPACE_SIZE = eval_env.action_space.n
    frame_stacker_eval = FrameStack(k=4) # Assuming FRAMES_PER_STATE is 4
    
    temp_obs_eval, _ = eval_env.reset()
    temp_state_eval = frame_stacker_eval.reset(temp_obs_eval)
    STATE_SHAPE_EVAL = temp_state_eval.shape

    agent_eval = DQNAgent(
        state_shape=STATE_SHAPE_EVAL,
        action_space=ACTION_SPACE_SIZE 
        # Other params like LR, gamma don't matter for eval if not training
    )
    
    if os.path.exists(model_path_to_load):
        print(f"Loading model for evaluation from {model_path_to_load}")
        agent_eval.load(model_path_to_load)
    else:
        print(f"ERROR: Model not found at {model_path_to_load}. Cannot evaluate.")
        eval_env.close()
        return -float('inf')

    agent_eval.policy_net.eval() # Set to evaluation mode

    total_rewards = []
    for episode in range(num_episodes):
        obs, info = eval_env.reset(seed=random.randint(0, 1_000_000)) # Different seed for each eval episode
        state = frame_stacker_eval.reset(obs)
        done = False
        episode_reward = 0
        steps = 0
        while not done:
            action = agent_eval.act(state, explore=False) # Act greedily
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            state = frame_stacker_eval.step(next_obs)
            episode_reward += reward
            steps +=1
            if human_render: # Check human_render flag
                eval_env.render() # Render if in human mode
            if steps > 20000: # Safety break for very long episodes
                print("Warning: Evaluation episode exceeded 20000 steps.")
                break
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}")

    eval_env.close()
    avg_reward = np.mean(total_rewards)
    print(f"Average evaluation reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

# --- Training Function ---
def train_pong_agent(human_render_during_training=False, load_checkpoint_flag=False):
    # Hyperparameters from "Smart Defaults"
    ENV_NAME = "PongNoFrameskip-v4"
    train_render_mode = "human" if human_render_during_training else None # Use None if not human rendering
    LEARNING_RATE = 1e-4 # Changed from 2.5e-4
    BATCH_SIZE = 32 # Defined inside train_pong_agent
    REPLAY_BUFFER_SIZE = int(1e5) # Defined inside train_pong_agent
    GAMMA = 0.99 # Defined inside train_pong_agent
    TARGET_UPDATE_FREQ = 1000 # Agent steps, defined inside train_pong_agent
    EPSILON_START = 1.0 # Defined inside train_pong_agent
    EPSILON_END = 0.01 # Defined inside train_pong_agent
    EPSILON_DECAY_FRAMES = int(1e6)

    MAX_EPISODES = 5000
    MAX_FRAMES_TOTAL = int(2e6)
    EVAL_INTERVAL_EPISODES = 10 # Changed from 100
    SAVE_INTERVAL_EPISODES = 10 # Changed from 500
    FRAMES_PER_STATE = 4

    # Construct path relative to the script's location to ensure it's project root based
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pong")
    os.makedirs(DATA_DIR, exist_ok=True)
    MODEL_PATH = os.path.join(DATA_DIR, "pong_dqn_model.pth") # Regular checkpoint
    BEST_MODEL_PATH = os.path.join(DATA_DIR, "pong_dqn_best_model.pth")
    REWARDS_PLOT_PATH = os.path.join(DATA_DIR, "pong_training_rewards.png")
    STATS_PATH = os.path.join(DATA_DIR, "pong_training_stats.json") # For resumable stats

    # Seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type != 'cpu':
        torch.cuda.manual_seed_all(SEED)

    env = gym.make(ENV_NAME, render_mode=train_render_mode, repeat_action_probability=0.0)
    # Important: Get the actual action space size from the environment
    ACTION_SPACE_SIZE = env.action_space.n
    print(f"Action space size: {ACTION_SPACE_SIZE}")

    # Initialize FrameStacker and Agent
    frame_stacker = FrameStack(k=FRAMES_PER_STATE) # Defined inside train_pong_agent
    # Initial observation to get state shape
    temp_obs, _ = env.reset(seed=SEED) # Defined inside train_pong_agent
    temp_state = frame_stacker.reset(temp_obs) # Defined inside train_pong_agent
    STATE_SHAPE = temp_state.shape # Defined inside train_pong_agent

    agent = DQNAgent( # Defined inside train_pong_agent
        state_shape=STATE_SHAPE, # Defined inside train_pong_agent
        action_space=ACTION_SPACE_SIZE, # Defined inside train_pong_agent
        lr=LEARNING_RATE, # Defined inside train_pong_agent
        gamma=GAMMA, # Defined inside train_pong_agent
        target_update_freq=TARGET_UPDATE_FREQ, # Defined inside train_pong_agent
        epsilon_start=EPSILON_START, # Defined inside train_pong_agent
        epsilon_end=EPSILON_END, # Defined inside train_pong_agent
        epsilon_decay_frames=EPSILON_DECAY_FRAMES, # Defined inside train_pong_agent
        batch_size=BATCH_SIZE # Defined inside train_pong_agent
    )
    
    if load_checkpoint_flag and os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        agent.load(MODEL_PATH)
    
    # Initialize stats
    all_episode_stats = [] # List of dicts, one per episode
    start_episode = 1
    cumulative_training_time_seconds_loaded = 0.0
    best_eval_reward = -float('inf') # Will be loaded from stats if available

    if load_checkpoint_flag:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model checkpoint from {MODEL_PATH}")
            agent.load(MODEL_PATH) # Loads weights and sets agent.current_frames
        else:
            print(f"Checkpoint {MODEL_PATH} not found. Starting new training.")
        
        if os.path.exists(STATS_PATH):
            print(f"Loading training stats from {STATS_PATH}")
            try:
                with open(STATS_PATH, 'r') as f:
                    stats = json.load(f)
                all_episode_stats = stats.get("episode_stats", [])
                # Restore current_frames from stats. Default to 0 if not found or if stats file is new.
                agent.current_frames = stats.get("total_agent_steps_completed", 0) 
                start_episode = stats.get("last_completed_episode_number", 0) + 1
                best_eval_reward = stats.get("best_eval_reward_achieved", -float('inf'))
                cumulative_training_time_seconds_loaded = stats.get("cumulative_training_time_seconds", 0.0)
                print(f"Resuming from episode {start_episode}, total steps: {agent.current_frames}, best_eval_reward: {best_eval_reward:.2f}, cumulative_time: {format_time(cumulative_training_time_seconds_loaded)}")
            except json.JSONDecodeError:
                print(f"Error decoding {STATS_PATH}. Starting with fresh stats.")
        else:
            print(f"Stats file {STATS_PATH} not found. Starting with fresh stats.")

    replay_buffer = ReplayBuffer(size=REPLAY_BUFFER_SIZE)
    
    # For plotting avg rewards later
    # We'll derive all_episode_rewards from all_episode_stats for plotting
    
    # --- Replay Buffer Warmup ---
    WARMUP_STEPS = 50000 # Number of steps to warm up the buffer
    if not load_checkpoint_flag or agent.current_frames < WARMUP_STEPS : # Only warmup if not resuming a sufficiently trained agent or if current frames are less than warmup
        print(f"Warming up replay buffer with random policy for {WARMUP_STEPS} steps...")
        obs_warmup, _ = env.reset(seed=SEED -1) # Use a different seed for warmup
        state_warmup = frame_stacker.reset(obs_warmup)
        frames_done_warmup = 0
        while frames_done_warmup < WARMUP_STEPS:
            action_warmup = env.action_space.sample()
            next_obs_warmup, reward_warmup, terminated_warmup, truncated_warmup, _ = env.step(action_warmup)
            done_warmup = terminated_warmup or truncated_warmup
            
            next_state_warmup = frame_stacker.step(next_obs_warmup)
            clipped_reward_warmup = np.sign(reward_warmup) # Reward Clipping
            replay_buffer.store((state_warmup, action_warmup, clipped_reward_warmup, next_state_warmup, float(done_warmup)))
            
            if done_warmup:
                obs_warmup, _ = env.reset(seed=SEED + frames_done_warmup) # Vary seed on reset, ensure positive
                state_warmup = frame_stacker.reset(obs_warmup)
            else:
                state_warmup = next_state_warmup
            frames_done_warmup +=1
            if frames_done_warmup % 1000 == 0:
                print(f"Warmup step {frames_done_warmup}/{WARMUP_STEPS}")
        print("Replay buffer warmup complete.")
        # Reset agent's current_frames if warmup happened for a new training run,
        # as act() increments it and warmup shouldn't count towards epsilon decay of main training.
        # However, if loading a checkpoint, agent.current_frames is already set from stats.
        # The warmup logic above is primarily for a fresh start or if a loaded agent has very few frames.
        # If agent.current_frames was loaded from stats and is > WARMUP_STEPS, warmup is skipped.
        # If agent.current_frames was loaded and < WARMUP_STEPS, it continues from there.
        # If fresh start (agent.current_frames = 0), it does full warmup.
        # The agent.act() calls during main training will correctly use the loaded/progressed current_frames.

    session_start_time = time.time()

    print(f"Starting training from episode {start_episode} up to {MAX_EPISODES} or until {MAX_FRAMES_TOTAL} agent steps...")
    if human_render_during_training:
        print("Human rendering enabled during training. This will be slower.")

    for episode_num in range(start_episode, MAX_EPISODES + 1):
        episode_start_time = time.time()
        obs, info = env.reset(seed=SEED + episode_num)
        state = frame_stacker.reset(obs)
        done = False
        current_episode_reward = 0 # Raw reward for this episode for logging
        current_episode_steps = 0

        while not done:
            action = agent.act(state) # agent.current_frames is incremented here
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if human_render_during_training:
                env.render()

            next_state = frame_stacker.step(next_obs)
            clipped_reward = np.sign(reward) # Reward Clipping
            replay_buffer.store((state, action, clipped_reward, next_state, float(done)))

            state = next_state
            current_episode_reward += reward # Log raw reward
            current_episode_steps += 1
            
            loss = agent.learn(replay_buffer) # agent.current_frames is incremented inside act()

            if agent.current_frames >= MAX_FRAMES_TOTAL:
                print(f"Reached max total agent steps ({MAX_FRAMES_TOTAL}). Stopping training.")
                done = True
        
        episode_duration_seconds = time.time() - episode_start_time
        current_session_active_time = time.time() - session_start_time
        total_cumulative_time_seconds = cumulative_training_time_seconds_loaded + current_session_active_time

        episode_data = {
            "episode_number": episode_num,
            "reward": current_episode_reward,
            "steps": current_episode_steps,
            "epsilon": agent.get_epsilon(),
            "timestamp_end_episode": time.time(),
            "duration_episode_seconds": episode_duration_seconds
        }
        all_episode_stats.append(episode_data)
        
        # For plotting convenience
        rewards_for_plot = [e['reward'] for e in all_episode_stats]
        avg_reward_for_plot = np.mean(rewards_for_plot[-100:])
        
        print(f"Episode: {episode_num}/{MAX_EPISODES} | Steps: {current_episode_steps} | Total Steps: {agent.current_frames} | Reward: {current_episode_reward:.2f} | Avg Reward (100ep): {avg_reward_for_plot:.2f} | Epsilon: {agent.get_epsilon():.4f} | Cum. Time: {format_time(total_cumulative_time_seconds)}")

        if episode_num % EVAL_INTERVAL_EPISODES == 0:
            current_model_to_eval_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
            # Ensure a model is saved before first evaluation if no best model exists yet
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
            
            # Plotting uses rewards_for_plot and its derived avg_reward_for_plot
            plot_rewards(rewards_for_plot, [np.mean(rewards_for_plot[max(0,i-99):i+1]) for i in range(len(rewards_for_plot))], filename=REWARDS_PLOT_PATH)


        if episode_num % SAVE_INTERVAL_EPISODES == 0:
            agent.save(MODEL_PATH)
            print(f"Model checkpoint saved at episode {episode_num} to {MODEL_PATH}")
            # Save stats with checkpoint
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

        if agent.current_frames >= MAX_FRAMES_TOTAL:
            break

    env.close()
    print("Training finished.")
    # Final save of model and stats
    agent.save(MODEL_PATH)
    final_total_cumulative_time = cumulative_training_time_seconds_loaded + (time.time() - session_start_time)
    final_stats_to_save = {
        "episode_stats": all_episode_stats,
        "total_agent_steps_completed": agent.current_frames,
        "last_completed_episode_number": MAX_EPISODES if agent.current_frames < MAX_FRAMES_TOTAL else episode_num, # Save last completed episode
        "best_eval_reward_achieved": best_eval_reward,
        "cumulative_training_time_seconds": final_total_cumulative_time
    }
    with open(STATS_PATH, 'w') as f:
        json.dump(final_stats_to_save, f, indent=4)
    print(f"Final model saved to {MODEL_PATH}")
    print(f"Final training stats saved to {STATS_PATH}")
    
    final_rewards_for_plot = [e['reward'] for e in all_episode_stats]
    final_avg_rewards_for_plot = [np.mean(final_rewards_for_plot[max(0,i-99):i+1]) for i in range(len(final_rewards_for_plot))]
    plot_rewards(final_rewards_for_plot, final_avg_rewards_for_plot, filename=REWARDS_PLOT_PATH)
    print(f"Final rewards plot saved to {REWARDS_PLOT_PATH}")
    return agent


if __name__ == "__main__":
    # Check for ALE
    if not 'ale_py' in globals() or not hasattr(ale_py, '__version__'):
        print("ERROR: ale_py not properly imported or available. Please install with:")
        print("pip install ale-py")
        exit(1)
    
    gym.register_envs(ale_py) # Register ALE environments

    # Check for Pong ROM (by trying to make the env)
    try:
        temp_env_check = gym.make("PongNoFrameskip-v4")
        temp_env_check.close()
        print("Pong ROM successfully loaded via ale_py.")
    except Exception as e:
        print(f"Error loading PongNoFrameskip-v4: {e}")
        print("This might be due to missing ROM files if ale_py is installed but ROMs are not.")
        print("If needed, you might need to install ROMs, e.g., `pip install gymnasium[accept-rom-license]` (use quotes in zsh).")
        exit(1)

    # Construct path relative to the script's location to ensure it's project root based
    SCRIPT_DIR_MAIN = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT_MAIN = os.path.dirname(SCRIPT_DIR_MAIN)
    DATA_DIR_MAIN = os.path.join(PROJECT_ROOT_MAIN, "data", "pong")
    MODEL_PATH_MAIN = os.path.join(DATA_DIR_MAIN, "pong_dqn_model.pth")
    BEST_MODEL_PATH_MAIN = os.path.join(DATA_DIR_MAIN, "pong_dqn_best_model.pth")


    while True:
        choice = input("Do you want to [t]rain or [e]valuate? ").lower()
        if choice in ['t', 'train']:
            render_train_choice = input("Enable human rendering during training? [y/n]: ").lower()
            do_human_render_train = render_train_choice in ['y', 'yes']
            
            load_choice = input("Load existing checkpoint to continue training? [y/n]: ").lower()
            load_checkpoint_train = load_choice in ['y', 'yes']
            
            print("\nStarting training...")
            train_pong_agent(human_render_during_training=do_human_render_train, load_checkpoint_flag=load_checkpoint_train)
            break
        elif choice in ['e', 'evaluate']:
            render_eval_choice = input("Enable human rendering during evaluation? [y/n]: ").lower()
            do_human_render_eval = render_eval_choice in ['y', 'yes']
            
            model_to_eval = BEST_MODEL_PATH_MAIN if os.path.exists(BEST_MODEL_PATH_MAIN) else MODEL_PATH_MAIN
            if not os.path.exists(model_to_eval):
                print(f"No model found at {model_to_eval} or {BEST_MODEL_PATH_MAIN}. Please train a model first.")
                break 

            print(f"\nEvaluating agent using model: {model_to_eval}...")
            evaluate_pong_agent("PongNoFrameskip-v4", model_to_eval, num_episodes=10, human_render=do_human_render_eval)
            break
        else:
            print("Invalid choice. Please enter 't' for train or 'e' for evaluate.")
