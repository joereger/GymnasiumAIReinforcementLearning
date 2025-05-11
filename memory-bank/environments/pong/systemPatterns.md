# System Patterns: Pong (`ALE/PongNoFrameskip-v4`)

This document outlines key code patterns, architectural choices, and reusable components specific to the Pong environment solution, based on the "Smart Defaults" provided.

## 1. Code Organization

The implementation has been refactored into a modular structure to improve maintainability and readability:

```
pong/
├── pong_dqn.py              # Main entry point
├── pong_dqn_utils.py        # Utilities (preprocessing, frame stacking, replay buffer)
├── pong_dqn_model.py        # Neural network model and agent implementation
├── pong_dqn_vis.py          # Visualization and plotting
└── pong_dqn_train.py        # Training and evaluation logic
```

This modular approach allows for:
- Better separation of concerns
- Easier maintenance and updates
- Reduced file sizes to prevent syntax issues
- Clearer documentation of each component's purpose

## 2. Environment Instantiation

The environment is instantiated as follows, ensuring manual control over frame skipping and correct naming:

```python
import gymnasium as gym
import ale_py # Required for ALE environments

# gym.register_envs(ale_py) # Called once, typically at the start of the script
# env = gym.make("PongNoFrameskip-v4", repeat_action_probability=0.0, render_mode=train_render_mode)
```
*Note: `PongNoFrameskip-v4` is the correct ID, not prefixed with `ALE/` for this variant. `ale_py` registration is needed.*

## 3. Input Preprocessing (in `pong_dqn_utils.py`)

A standard preprocessing function is used for each frame:

```python
import cv2
import numpy as np

def preprocess(frame):
    """
    Preprocesses a single frame from the Pong environment.
    - Converts to grayscale
    - Downsamples to 84x84
    - Normalizes pixel values to [0, 1]
    """
    if frame.ndim == 3 and frame.shape[-1] == 3: # Check if it's an RGB frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # frame might already be grayscale if coming from a wrapper or specific env mode
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = frame.astype(np.float32) / 255.0
    return frame
```

## 4. Frame Stacking (in `pong_dqn_utils.py`)

A `FrameStack` class is used to maintain a history of the last `k` (typically 4) preprocessed frames, providing temporal context to the agent:

```python
from collections import deque
import numpy as np

class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self, obs):
        """
        Resets the frame stack with the initial observation.
        The initial observation is preprocessed and duplicated k times.
        """
        processed_obs = preprocess(obs)
        for _ in range(self.k):
            self.frames.append(processed_obs)
        return np.stack(list(self.frames), axis=0)

    def step(self, obs):
        """
        Appends the new preprocessed observation to the stack and returns the stack.
        """
        self.frames.append(preprocess(obs))
        return np.stack(list(self.frames), axis=0)
```

## 5. DQN Model Architecture (in `pong_dqn_model.py`)

The standard DeepMind 2015 Atari CNN architecture is implemented using PyTorch:

```python
import torch
import torch.nn as nn

class PongDQN(nn.Module):
    def __init__(self, input_channels=4, action_space=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        # For input (4, 84, 84):
        # Conv1: (84-8)/4 + 1 = 20 -> (32, 20, 20)
        # Conv2: (20-4)/2 + 1 = 9  -> (64, 9, 9)
        # Conv3: (9-3)/1 + 1 = 7   -> (64, 7, 7)
        self.fc_input_dims = 7 * 7 * 64

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512), nn.ReLU(inplace=True),
            nn.Linear(512, action_space),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)
```

## 6. DQN Agent (in `pong_dqn_model.py`)

The agent handles:
- Epsilon-greedy policy with exponential decay
- Network parameter updates 
- Target network updates
- Model saving and loading

```python
class DQNAgent:
    def __init__(self, state_shape, action_space, lr=2.5e-4, gamma=0.99, 
                 target_update_freq=10000, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay_frames=1e6, batch_size=32):
        # Initialize networks, optimizer and parameters
        # ...

    def get_epsilon(self):
        # Calculate current epsilon based on frames seen
        # ...

    def act(self, state, explore=True):
        # Select action using epsilon-greedy
        # ...

    def learn(self, replay_buffer):
        # Update network from experiences
        # Apply gradient clipping
        # Periodically update target network
        # ...

    def save(self, path):
        # Save model weights
        # ...

    def load(self, path):
        # Load model weights and update target network
        # ...
```

## 7. Replay Buffer (in `pong_dqn_utils.py`)

A simple deque-based replay buffer:

```python
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
```
*Note: Replay buffer stores transitions with **clipped rewards** (`np.sign(reward)`).*

## 8. Visualization (in `pong_dqn_vis.py`)

The `plot_training_data` function now creates a single figure with multiple y-axes to display different metrics:

```python
def plot_training_data(episode_stats, filename="pong_training_progress.png"):
    # Create figure with primary y-axis for rewards
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Plot rewards on primary y-axis
    ax1.plot(episodes, rewards, label='Episode Reward')
    ax1.plot(episodes, avg_rewards, label='Avg Reward (100ep)', color='red')
    
    # Add secondary y-axis for avg_max_q if available
    if has_avg_max_q:
        ax2 = ax1.twinx()
        ax2.plot(episodes, avg_max_q_values, label='Avg Max Q', color='green')
    
    # Add tertiary y-axis for avg_loss if available
    if has_avg_loss:
        ax3 = ax1.twinx()
        if ax2 is not None:
            ax3.spines["right"].set_position(("outward", 60))
        ax3.plot(episodes, avg_loss_values, label='Avg Loss', color='purple')
    
    # Add legend, save, etc.
    # ...
```

This approach provides a comprehensive view of all metrics on a single plot with properly scaled axes.

## 9. Training and Evaluation Logic (in `pong_dqn_train.py`) 

The training and evaluation functions provide the main logic for the DQN algorithm:

- **Training function:** Manages the main training loop with:
  - Replay buffer warmup
  - Epsilon-greedy exploration
  - Experience collection and network updates
  - Periodic evaluation and checkpointing
  - Detailed logging and statistics tracking

- **Evaluation function:** Provides greedy evaluation of the trained agent with:
  - No exploration (epsilon=0)
  - Optional rendering for visual inspection
  - Average reward calculation across multiple episodes

## 10. Key Training Enhancements

The implementation includes several enhancements to the standard DQN algorithm:

- **Reward Clipping:** Raw rewards from the environment are clipped to `np.sign(reward)` (i.e., -1, 0, or 1) before being stored in the replay buffer and used for learning. This helps stabilize Q-value updates.

- **Replay Buffer Warmup:** Before the main training loop starts, the replay buffer is pre-filled with `50,000` experiences generated by a random policy. This ensures the agent learns from more diverse initial samples.

- **Gradient Clipping:** During the `agent.learn()` step, gradients are clipped to a maximum norm of `1.0` (`torch.nn.utils.clip_grad_norm_`) to prevent exploding gradients and further stabilize training.

- **Experiment 1 Settings:**
  - Learning Rate: `2.5e-4`
  - Target Network Update Frequency: `10,000` steps (increased from 1,000)

## 11. Logging and Statistics

The implementation tracks and logs detailed statistics:

- **Per-episode metrics:**
  - Raw reward sum
  - Number of steps
  - Epsilon value
  - Average loss 
  - Average maximum Q-value 
  - Episode duration

- **Global metrics:**
  - Total agent steps
  - Best evaluation reward
  - Cumulative training time

All statistics are saved to `data/pong/pong_training_stats.json` for resumable training and visualization.

This modular and enhanced implementation provides a robust framework for experimenting with DQN on the Pong environment while maintaining good software engineering practices.
