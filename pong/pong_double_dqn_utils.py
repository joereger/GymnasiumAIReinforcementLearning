import numpy as np
import cv2
from collections import deque
import random
import time

# --- Preprocessing ---
def preprocess(frame):
    """Preprocess a frame for input to the DQN.
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

# --- Replay Buffer ---
class ReplayBuffer:
    """Store and sample transitions for DQN training.
    
    For Double DQN, the replay buffer implementation remains the same,
    but we use a larger buffer size (500K vs 100K) to provide more diverse
    experiences and enhance stability.
    """
    def __init__(self, size=int(5e5)):  # Increased from 1e5 to 5e5 for Experiment 2
        self.buffer = deque(maxlen=size)

    def store(self, transition):
        """Store a transition (state, action, reward, next_state, done)."""
        self.buffer.append(transition)

    def sample(self, batch_size=32):
        """Sample a batch of transitions randomly."""
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

# --- Helper to format time ---
def format_time(seconds):
    """Format seconds into a human-readable HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
