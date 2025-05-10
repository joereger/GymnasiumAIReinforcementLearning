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

# Create data directory if it doesn't exist
os.makedirs("data/freeway", exist_ok=True)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class FrameSkip(gym.Wrapper):
    """
    Repeat the same action for k frames and return the max of the last 2 frames.
    Standard preprocessing for Atari games.
    """
    def __init__(self, env, skip=4):
        super(FrameSkip, self).__init__(env)
        self.skip = skip
        self.frames = deque(maxlen=2)  # For max pooling over the last 2 frames
    
    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        
        # Repeat action for skip frames
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.frames.append(observation)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        # Max pooling over the last 2 frames to reduce flickering
        if len(self.frames) == 1:
            max_frame = self.frames[0]
        else:
            max_frame = np.maximum(self.frames[-1], self.frames[-2])
        
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.frames.clear()
        self.frames.append(observation)
        return observation, info


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


class DQNNetwork(nn.Module):
    """
    Standard Deep Q-Network for Atari games (3 conv layers + 1 FC layer)
    Based on the original DQN paper architecture.
    """
    def __init__(self, input_shape, n_actions):
        super(DQNNetwork, self).__init__()
        self.input_shape = input_shape
        
        # Convolutional layers (standard architecture)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate size after convolutions (needed for FC layers)
        conv_output_size = self._get_conv_output(input_shape)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output(self, shape):
        """Calculate output size after convolution layers."""
        o = torch.zeros(1, *shape)
        o = self.conv_layers(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        return self.fc_layers(x)


class ReplayBuffer:
    """Large experience replay buffer for DQN."""
    def __init__(self, capacity=100000):
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


class DoubleDQNAgent:
    """Double Deep Q-Network agent for playing Freeway."""
    def __init__(self, state_shape, n_actions, learning_rate=0.00025, gamma=0.99, buffer_size=100000):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.gamma = gamma
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_shape, n_actions).to(device)
        self.target_net = DQNNetwork(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer with the recommended learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999  # Slower decay for longer exploration
        self.total_frames = 0
        self.epsilon_decay_frames = 500000  # Linear decay over 500k frames
        
        # Training parameters
        self.batch_size = 32
        self.target_update_freq = 1000
        self.train_steps = 0
        
        # Metrics
        self.loss = 0
        self.max_q = 0
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy."""
        # Linear epsilon decay based on frames
        if self.total_frames < self.epsilon_decay_frames:
            self.epsilon = 1.0 - 0.99 * (self.total_frames / self.epsilon_decay_frames)
        else:
            self.epsilon = 0.01
        
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
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_frames += 1
    
    def train(self):
        """Train the network with a batch from replay buffer (Double DQN)."""
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
        
        # Compute next Q-values using Double DQN
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            # Get Q-values from target network for those actions
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            # Compute target Q-values
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
            'total_frames': self.total_frames,
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
            if 'total_frames' in checkpoint:
                self.total_frames = checkpoint['total_frames']
            print(f"Model loaded from {filename}")
        else:
            print(f"No model found at {filename}")


def train_agent(episodes=10000, render=True, load_checkpoint=False):
    """Train the Double DQN agent on Freeway."""
    # Create environment with preprocessing and frame skipping
    env_id = "ALE/Freeway-v5"
    # Use human mode for visualization if specified
    render_mode = "human" if render else None
    
    # Create base environment
    base_env = gym.make(
        env_id, 
        render_mode=render_mode,
        repeat_action_probability=0.0,  # Deterministic environment
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
    print("- Time limit for our episodes: 4800 frames (1200 steps with frame skip)")

    # Apply frame skipping, then preprocessing
    skip_env = FrameSkip(base_env, skip=4)
    env = PreprocessFreeway(skip_env)
    print("Using frame skip (4) and the standard preprocessing pipeline")
    
    # Initialize Double DQN agent
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DoubleDQNAgent(state_shape, n_actions)
    print("Using Double DQN with 3-layer CNN architecture")
    
    # Load checkpoint if requested
    if load_checkpoint:
        checkpoint_path = "data/freeway/freeway_double_dqn.pth"
        agent.load(checkpoint_path)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    episode_max_qs = []
    best_reward = float('-inf')
    start_time = time.time()
    
    # Max number of steps per episode (with frame skip, this is 4800 actual frames)
    max_steps = 1200
    
    # Training loop
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_max_q = 0
        
        # Episode loop
        while True:
            # Select action using epsilon-greedy
            action = agent.select_action(state)
                
            # Print occasional debugging info
            if episode_length % 300 == 0:  # Log every 300 steps (1200 frames)
                print(f"  Step {episode_length} (Frame {agent.total_frames}): Action={action} ({['NOOP', 'UP', 'DOWN'][action]}), Epsilon={agent.epsilon:.3f}")
            
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
            
            # Ensure episodes don't run forever
            if done or episode_length >= max_steps:
                if episode_length >= max_steps:
                    print(f"  Episode forcibly terminated at {episode_length} steps ({episode_length * 4} frames)")
                break
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_max_qs.append(episode_max_q)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("data/freeway/freeway_double_dqn_best.pth")
        
        # Save checkpoint periodically
        if episode % 50 == 0:
            agent.save("data/freeway/freeway_double_dqn.pth")
            # Save training curves
            plot_training_progress(episode_rewards, episode_lengths, episode_max_qs)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        
        # Single-line logging with frame count
        print(f"Episode: {episode} | Score: {episode_reward:.2f} | Steps: {episode_length} | Frames: {agent.total_frames} | Epsilon: {agent.epsilon:.3f} | Loss: {agent.loss:.4f} | Max Q: {episode_max_q:.2f} | Memory: {len(agent.replay_buffer)} | Time: {elapsed_str}")
    
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
    plt.savefig('data/freeway/training_progress_double_dqn.png')
    plt.close()


def evaluate_agent(episodes=10, render=True):
    """Evaluate a trained agent without exploration."""
    # Load environment
    env_id = "ALE/Freeway-v5"
    # Use human mode for visual rendering
    render_mode = "human" if render else None
    
    # Create base environment
    base_env = gym.make(
        env_id, 
        render_mode=render_mode,
        repeat_action_probability=0.0,
        full_action_space=False,
        disable_env_checker=True
    )
    
    # Apply wrappers
    skip_env = FrameSkip(base_env, skip=4)
    env = PreprocessFreeway(skip_env)
    
    # Print notice about rendering
    if render:
        print("Using human render mode for evaluation (there might be sound)")
    
    # Initialize agent
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    agent = DoubleDQNAgent(state_shape, n_actions)
    
    # Load best model
    agent.load("data/freeway/freeway_double_dqn_best.pth")
    agent.epsilon = 0.0  # No exploration during evaluation
    
    # Evaluation metrics
    total_rewards = []
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        step = 0
        
        while True:
            step += 1
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            
            if done or step >= 1200:  # Same max steps as training
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
        env = gym.make(
            "ALE/Freeway-v5", 
            render_mode=None,
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
