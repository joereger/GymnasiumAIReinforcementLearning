import os
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
from datetime import datetime

# Import and register Atari environments
try:
    import ale_py
    gym.register_envs(ale_py)
    atari_available = True
except ImportError:
    atari_available = False
    print("Warning: ale_py not found, Atari environments will not be available")
    print("Install with: pip install ale-py")

# Get project root directory (regardless of where script is executed from)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Create data directory if it doesn't exist
data_dir = os.path.join(project_root, "data", "freeway")
os.makedirs(data_dir, exist_ok=True)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PreprocessFreeway(gym.Wrapper):
    """
    Preprocessing wrapper for Atari Freeway:
    1. Convert RGB to grayscale
    2. Resize to 84x84
    3. Stack 4 frames
    4. Normalize pixel values
    """
    def __init__(self, env, frame_stack=4, resize_shape=(84, 84)):
        super(PreprocessFreeway, self).__init__(env)
        self.frame_stack = frame_stack
        self.resize_shape = resize_shape
        self.frames = deque(maxlen=frame_stack)
        
        # Update observation space for stacked grayscale frames
        self.observation_space = spaces.Box(
            low=0, high=1.0, 
            shape=(frame_stack, resize_shape[0], resize_shape[1]),
            dtype=np.float32
        )
    
    def preprocess_frame(self, frame):
        """Convert RGB frame to grayscale, resize, and normalize."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize
        resized = cv2.resize(gray, self.resize_shape, interpolation=cv2.INTER_AREA)
        # Normalize
        normalized = resized / 255.0
        return normalized

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        processed_frame = self.preprocess_frame(observation)
        
        # Initialize frame stack with the first frame
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
            
        # Stack frames into a single observation
        stacked_frames = np.array(self.frames, dtype=np.float32)
        return stacked_frames, info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        processed_frame = self.preprocess_frame(observation)
        
        # Add new frame to stack
        self.frames.append(processed_frame)
        
        # Stack frames into a single observation
        stacked_frames = np.array(self.frames, dtype=np.float32)
        
        return stacked_frames, reward, terminated, truncated, info


class CustomRewardFreeway(gym.Wrapper):
    """
    Custom reward wrapper for Freeway to provide more learning signals:
    1. Original reward: +1 for reaching the top
    2. Progress reward: Small reward for moving upward
    3. Penalty for moving backward
    4. Time penalty to encourage efficiency
    """
    def __init__(self, env):
        super(CustomRewardFreeway, self).__init__(env)
        # Original observation is 210x160, with the chicken in the bottom portion
        # We'll track the chicken's vertical position to award progress rewards
        self.prev_y_position = None
        self.screen_height = 210
        
        # Define zones for progress rewards (divide screen into 10 vertical zones)
        self.num_zones = 10
        self.zone_height = self.screen_height / self.num_zones
        self.current_zone = 0
        
        # Find correct RAM address for player position
        self.player_pos_address = self._find_player_position_address()
        print(f"Player position RAM address: {self.player_pos_address}")
    
    def _find_player_position_address(self):
        """
        Find the correct RAM address for the player's position.
        For Freeway, we use a safe address within known RAM boundaries.
        """
        # For Freeway, we'll use a default address that's safely within bounds
        # of the known RAM size (128)
        return 0x10  # (16 in decimal) - Well within RAM bounds
    
    def _debug_ram(self, ram):
        """Debug function to print RAM ranges that might contain player position."""
        print("RAM Analysis (potential player position addresses):")
        # Print RAM size and some bounds checking
        ram_size = len(ram)
        print(f"Total RAM size: {ram_size}")
        print(f"Valid indices: 0 to {ram_size-1}")
        
        # Print some sample ranges - being extremely careful with bounds
        print("\nSample RAM values (safe access):")
        # Sample from different parts of RAM to find potential patterns
        safe_ranges = [(0, 10), (10, 20), (30, 40), (60, 70), (100, 110)]
        
        for start, end in safe_ranges:
            print(f"\nRange 0x{start:02x}-0x{min(end, ram_size-1):02x}:")
            for addr in range(start, min(end, ram_size)):
                try:
                    print(f"RAM[0x{addr:02x}] = {ram[addr]}")
                except IndexError:
                    print(f"RAM[0x{addr:02x}] = <out of bounds>")
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Get RAM and initialize position tracking
        ram = self.env.unwrapped.ale.getRAM()
        
        # For first reset, debug RAM values to help find player position
        if self.prev_y_position is None:
            self._debug_ram(ram)
        
        # Get player position, with fallback if address is invalid
        try:
            self.prev_y_position = self.estimate_y_position(ram)
        except IndexError:
            print(f"Warning: Could not access RAM address 0x{self.player_pos_address:02x}")
            print("Falling back to default position value")
            self.prev_y_position = 100  # Default fallback value
            
        self.current_zone = self.get_zone(self.prev_y_position)
        return obs, info
    
    def estimate_y_position(self, ram):
        """
        Extract the chicken's Y position from RAM.
        """
        ram_size = len(ram)
        
        # First verify the address is in bounds
        if self.player_pos_address >= ram_size:
            print(f"Warning: Address 0x{self.player_pos_address:02x} is out of bounds")
            # Try a series of common addresses, all well within RAM size
            for addr in [0x10, 0x20, 0x30, 0x40, 0x50]:
                if addr < ram_size:
                    self.player_pos_address = addr
                    print(f"Trying alternative address: 0x{addr:02x}")
                    break
        
        # Access RAM with safety check
        if self.player_pos_address < ram_size:
            try:
                return ram[self.player_pos_address]
            except IndexError:
                print(f"Error accessing RAM at 0x{self.player_pos_address:02x}")
        
        # Default fallback value if all else fails
        print("Using default position value")
        return 100
    
    def get_zone(self, y_position):
        """Determine which vertical zone the chicken is in."""
        # In Freeway, lower y values mean higher up on the screen
        # Adjust the zone calculation accordingly (higher zone number = higher up on screen)
        normalized_pos = 1.0 - (y_position / self.screen_height)
        zone = int(normalized_pos * self.num_zones)
        return min(max(0, zone), self.num_zones - 1)  # Clamp between 0 and num_zones-1
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get current position (with error handling)
        ram = self.env.unwrapped.ale.getRAM()
        try:
            current_y = self.estimate_y_position(ram)
        except IndexError:
            # If RAM access fails, try to use visual position estimation in future versions
            print("Warning: RAM position tracking failed. Using previous position.")
            current_y = self.prev_y_position
            
        current_zone = self.get_zone(current_y)
        
        # Calculate modified reward
        modified_reward = reward  # Start with original reward
        
        # Add progress reward for entering a new higher zone
        if current_zone > self.current_zone:
            zone_progress = current_zone - self.current_zone
            modified_reward += 0.1 * zone_progress
            self.current_zone = current_zone
        
        # Add movement rewards/penalties
        if current_y < self.prev_y_position:  # Moving upward (forward progress)
            progress = (self.prev_y_position - current_y) / self.screen_height
            modified_reward += 0.1 * progress
        elif current_y > self.prev_y_position:  # Moving downward (backward movement)
            regress = (current_y - self.prev_y_position) / self.screen_height
            modified_reward -= 0.1 * regress
        
        # Add small time penalty to encourage efficiency
        modified_reward -= 0.01
        
        # Update position tracking for next step
        self.prev_y_position = current_y
        
        # Add debugging info
        info['original_reward'] = reward
        info['modified_reward'] = modified_reward
        info['y_position'] = current_y
        info['zone'] = current_zone
        
        return obs, modified_reward, terminated, truncated, info


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Freeway with specified architecture:
    - 4 convolutional layers
    - 3 fully connected layers
    """
    def __init__(self, input_shape, n_actions):
        super(DQNNetwork, self).__init__()
        self.input_shape = input_shape
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        
        # Calculate size after convolutions (needed for FC layers)
        self._feature_size = self._get_conv_output(input_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)
    
    def _get_conv_output(self, shape):
        """Calculate output size after convolution layers."""
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        o = F.relu(self.conv4(o))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add a new experience to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for playing Freeway."""
    def __init__(self, state_shape, n_actions, learning_rate=1e-3, gamma=0.99, buffer_size=10000):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_shape, n_actions).to(device)
        self.target_net = DQNNetwork(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer with higher learning rate (1e-3 instead of 1e-4)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # Individual episodes still use 0.90 for faster decay
        
        # Training parameters
        self.batch_size = 64  # Increased from 32
        self.target_update_freq = 500  # Decreased from 1000 for faster learning
        self.train_steps = 0
        
        # Metrics
        self.loss = 0
        self.max_q = 0
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        if random.random() > self.epsilon:
            # Exploit: choose best action based on Q-values
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                self.max_q = q_values.max().item()  # Track max Q-value for logging
                return q_values.argmax().item()
        else:
            # Explore: choose random action
            return random.randrange(self.n_actions)
    
    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the network with a batch from replay buffer."""
        # Skip if not enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Compute current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(q_values.squeeze(), target_q_values)
        self.loss = loss.item()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        """Save model weights."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load model weights."""
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.train_steps = checkpoint['train_steps']
            print(f"Model loaded from {filename}")
        else:
            print(f"No model found at {filename}")


def train_agent(episodes=10000, render=True, load_checkpoint=False):
    """Train the DQN agent on Freeway."""
    # Create environment with preprocessing and custom rewards
    env_id = "ALE/Freeway-v5"
    # Use human mode but with no sound (even if there is sound, it's better than no visuals)
    render_mode = "human" if render else None
    # NOTE: In Freeway, collisions push the chicken back down rather than ending the episode
    base_env = gym.make(
        env_id, 
        render_mode=render_mode,
        repeat_action_probability=0.0,  # Deterministic environment
        frameskip=1,                    # Use default frame skip
        full_action_space=False,        # Only use meaningful actions
        disable_env_checker=True        # Disable environment checker warnings
    )
    
    # Print notice about rendering
    if render:
        print("Using human render mode (there might be sound)")
        print("If you want to disable sound, you may need to mute your system volume")

    # Print action space and game mechanics
    print(f"Action space: {base_env.action_space}")
    print(f"Action meanings: {base_env.unwrapped.get_action_meanings()}")
    print("\nFREEWAY GAME MECHANICS:")
    print("- Goal: Get the chicken across the road to the top of the screen")
    print("- When hit by cars, the chicken is pushed back down (not killed)")
    print("- Each successful crossing scores +1 point")
    print("- Episodes normally end after a time limit (not on collision)")
    print("- Note: Our implementation forces termination after 2000 steps\n")

    # Use basic environment without custom rewards for initial training
    env = PreprocessFreeway(base_env)
    print("Using original rewards for stable learning")
    
    # Initialize agent
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions)
    
    # Load checkpoint if requested
    if load_checkpoint:
        checkpoint_path = os.path.join(data_dir, "freeway_dqn.pth")
        agent.load(checkpoint_path)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_max_qs = []
    best_reward = float('-inf')
    start_time = time.time()
    
    # Training loop
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_max_q = 0
        
        # Episode loop
        while True:
            # Biasing toward UP action during exploration
            # In Freeway: 0=NOOP, 1=UP, 2=DOWN
            if random.random() < 0.75 and agent.epsilon > 0.3:  # 75% chance of UP during early exploration
                action = 1  # Force UP action to encourage crossing
            else:
                action = agent.select_action(state)
                
            # Print occasional debugging info
            if episode_length % 500 == 0:
                print(f"  Step {episode_length}: Action={action} ({['NOOP', 'UP', 'DOWN'][action]})")
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store the transition in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Move to the next state
            state = next_state
            episode_reward += reward
            episode_length += 1
            episode_max_q = max(episode_max_q, agent.max_q)
            
            # Train the agent
            agent.train()
            
            # Ensure episodes don't run forever (force termination after 2000 steps)
            if done or episode_length >= 2000:
                if episode_length >= 2000:
                    print(f"  Episode forcibly terminated at {episode_length} steps")
                break
        
        # Use an even more aggressive exploration decay to accelerate learning
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.90)
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_max_qs.append(episode_max_q)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_path = os.path.join(data_dir, "freeway_dqn_best.pth")
            agent.save(best_model_path)
        
        # Save checkpoint periodically
        if episode % 100 == 0:
            checkpoint_path = os.path.join(data_dir, "freeway_dqn.pth")
            agent.save(checkpoint_path)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        
        # Single-line logging
        print(f"Episode: {episode} | Score: {episode_reward:.2f} | Steps: {episode_length} | Epsilon: {agent.epsilon:.3f} | Loss: {agent.loss:.4f} | Max Q: {episode_max_q:.2f} | Memory: {len(agent.replay_buffer)} | Time: {elapsed_str}")
        
        # Save training curves periodically
        if episode % 100 == 0:
            plot_training_progress(episode_rewards, episode_lengths, episode_max_qs)
    
    # Close environment
    env.close()
    return agent


def plot_training_progress(rewards, lengths, max_qs):
    """Plot and save training progress curves."""
    plt.figure(figsize=(12, 10))
    
    # Plot rewards
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot episode lengths
    plt.subplot(3, 1, 2)
    plt.plot(lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    # Plot max Q-values
    plt.subplot(3, 1, 3)
    plt.plot(max_qs)
    plt.title('Max Q-Values')
    plt.xlabel('Episode')
    plt.ylabel('Max Q-Value')
    
    # Save figure
    plt.tight_layout()
    progress_path = os.path.join(data_dir, "training_progress.png")
    plt.savefig(progress_path)
    plt.close()


def evaluate_agent(episodes=10, render=True):
    """Evaluate a trained agent without exploration."""
    # Load environment
    env_id = "ALE/Freeway-v5"
    # Use human mode for visual rendering
    render_mode = "human" if render else None
    
    # Use the same environment settings as in training
    base_env = gym.make(
        env_id, 
        render_mode=render_mode,
        repeat_action_probability=0.0,
        frameskip=1,
        full_action_space=False,
        disable_env_checker=True
    )
    
    # Print notice about sound
    if render:
        print("Using human render mode for evaluation (there might be sound)")
    env = PreprocessFreeway(base_env)  # No custom rewards during evaluation
    
    # Initialize agent
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DQNAgent(state_shape, n_actions)
    
    # Load best model
    best_model_path = os.path.join(data_dir, "freeway_dqn_best.pth")
    agent.load(best_model_path)
    agent.epsilon = 0.0  # No exploration during evaluation
    
    # Evaluation metrics
    total_rewards = []
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode}: Reward = {episode_reward}")
    
    # Calculate statistics
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    
    print(f"\nEvaluation Results ({episodes} episodes):")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    
    env.close()
    return avg_reward, std_reward


if __name__ == "__main__":
    # Check for ALE and ROMs
    if not atari_available:
        print("ERROR: ale_py not available. Please install with:")
        print("pip install ale-py")
        exit(1)
        
    # Check for Freeway ROM
    try:
        # Basic environment check - no render_mode means no sound
        env = gym.make(
            "ALE/Freeway-v5", 
            render_mode=None,   # No visual rendering or sound
            disable_env_checker=True
        )
        env.close()
        print("Freeway ROM successfully loaded.")
    except Exception as e:
        print(f"Error loading Freeway ROM: {e}")
        print("This might be due to missing ROM files. If needed, install ROMs with:")
        print("pip install 'gymnasium[accept-rom-license]'")
        print("Note: Use quotes around the brackets in zsh shell")
        exit(1)
    
    # Create folders if they don't exist
    os.makedirs("data/freeway", exist_ok=True)
    
    # Ask if user wants to train or evaluate
    while True:
        choice = input("Do you want to [t]rain or [e]valuate? ").lower()
        if choice in ['t', 'train']:
            # Ask about rendering
            render_choice = input("Enable rendering during training? [y/n] ").lower()
            render = render_choice in ['y', 'yes']
            
            # Ask about loading checkpoint
            load_choice = input("Load existing checkpoint? [y/n] ").lower()
            load_checkpoint = load_choice in ['y', 'yes']
            
            # Start training
            print("\nStarting training...")
            agent = train_agent(episodes=10000, render=render, load_checkpoint=load_checkpoint)
            break
        elif choice in ['e', 'evaluate']:
            print("\nEvaluating agent...")
            evaluate_agent(episodes=10, render=True)
            break
        else:
            print("Invalid choice. Please enter 't' for train or 'e' for evaluate.")
