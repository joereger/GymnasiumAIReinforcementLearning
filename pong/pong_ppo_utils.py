import numpy as np
import cv2
import torch
import time
from collections import deque

# --- Preprocessing ---
def preprocess(frame):
    """Preprocess a frame for input to the neural networks.
    - Convert to grayscale
    - Resize to 84x84
    - Normalize pixel values to [0, 1]
    """
    if frame.ndim == 3 and frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame

# --- Frame Stacking ---
class FrameStack:
    """Stack k frames together to provide temporal context."""
    def __init__(self, k=4):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        """Reset the frame stack with the initial observation."""
        processed_obs = preprocess(obs)
        for _ in range(self.k):
            self.frames.append(processed_obs)
        return np.stack(list(self.frames), axis=0)

    def step(self, obs):
        """Add a new observation to the frame stack."""
        self.frames.append(preprocess(obs))
        return np.stack(list(self.frames), axis=0)

# --- Rollout Buffer ---
class RolloutBuffer:
    """Store transitions collected during PPO rollouts."""
    
    def __init__(self, buffer_size, state_shape, action_dim, device):
        """Initialize a rollout buffer for PPO.
        
        Args:
            buffer_size: Maximum number of timesteps to store in the buffer
            state_shape: Shape of states (e.g., (4, 84, 84) for stacked frames)
            action_dim: Number of possible actions
            device: PyTorch device
        """
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device
        
        # Initialize storage for rollout data
        self.states = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        
        # For GAE and returns calculation
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        # Pointer to current position in buffer
        self.idx = 0
        self.full = False
    
    def store(self, state, action, reward, done, log_prob, value):
        """Store a transition in the buffer."""
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.log_probs[self.idx] = log_prob
        self.values[self.idx] = value
        
        # Update pointer
        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True
    
    def compute_advantages_and_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation (GAE) and returns.
        
        Args:
            last_value: Value estimate for the state after the last state in the buffer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter for advantage weighting
        """
        # For PTB length - 1 to 0
        # Length of the buffer is self.buffer_size if full, else self.idx
        buffer_len = self.buffer_size if self.full else self.idx
        
        if buffer_len == 0:
            return
        
        # Handle potential partial buffer - work with the valid portion only
        # In the case we have a non-full buffer, actual buffer len is self.idx (last valid idx + 1)
        last_gae = 0
        for step in reversed(range(buffer_len)):
            # If we're at the last step, use last_value for the next value
            # Otherwise, use the value from the next step
            if step == buffer_len - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            # TD error: reward + gamma * next_value * (1 - done) - current_value
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            
            # GAE formula: sum_t (gamma * lambda)^t * delta_{t+k}
            self.advantages[step] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            
            # Returns = advantages + values
            self.returns[step] = self.advantages[step] + self.values[step]
    
    def get_batch(self, batch_size=None):
        """Get a batch of data from the buffer.
        
        If batch_size is None, returns the entire buffer.
        Otherwise, returns a random sample of size batch_size.
        
        Returns:
            Dict of torch tensors containing the batch data
        """
        # Length of the buffer is self.buffer_size if full, else self.idx
        buffer_len = self.buffer_size if self.full else self.idx
        
        if buffer_len == 0:
            return None
        
        if batch_size is None:
            # Return entire buffer
            indices = np.arange(buffer_len)
        else:
            # Sample random indices
            indices = np.random.choice(buffer_len, min(buffer_len, batch_size), replace=False)
        
        # Extract data
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        log_probs_old = torch.FloatTensor(self.log_probs[indices]).to(self.device)
        values = torch.FloatTensor(self.values[indices]).to(self.device)
        returns = torch.FloatTensor(self.returns[indices]).to(self.device)
        advantages = torch.FloatTensor(self.advantages[indices]).to(self.device)
        
        # Normalize advantages (helps with training stability)
        if len(advantages) > 1:
            # Only normalize if we have more than one sample
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Return as dict
        return {
            "states": states,
            "actions": actions,
            "log_probs_old": log_probs_old,
            "values": values,
            "returns": returns,
            "advantages": advantages
        }
    
    def clear(self):
        """Clear the buffer by resetting the pointer."""
        self.idx = 0
        self.full = False

# --- Helper to format time ---
def format_time(seconds):
    """Format seconds into a human-readable HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
