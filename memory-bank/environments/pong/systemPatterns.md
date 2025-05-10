# System Patterns: Pong (`ALE/PongNoFrameskip-v4`)

This document outlines key code patterns, architectural choices, and reusable components specific to the Pong environment solution, based on the "Smart Defaults" provided.

## 1. Environment Instantiation

The environment is instantiated as follows, ensuring manual control over frame skipping:

```python
import gymnasium as gym

# env = gym.make("ALE/PongNoFrameskip-v4", repeat_action_probability=0.0)
```

## 2. Input Preprocessing

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
*Note: Added a check for RGB frame before `cvtColor` as raw Atari frames are RGB.*

## 3. Frame Stacking

A `FrameStack` class is used to maintain a history of the last `k` (typically 4) preprocessed frames, providing temporal context to the agent.

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
        return np.stack(list(self.frames), axis=0) # Convert deque to list before stacking for older numpy

    def step(self, obs):
        """
        Appends the new preprocessed observation to the stack and returns the stack.
        """
        self.frames.append(preprocess(obs))
        return np.stack(list(self.frames), axis=0) # Convert deque to list before stacking
```
*Note: Modified `np.stack(self.frames, ...)` to `np.stack(list(self.frames), ...)` for broader compatibility as `np.stack` directly on a deque might behave differently or error on some NumPy versions.*

## 4. DQN Model Architecture (PyTorch)

The standard DeepMind 2015 Atari CNN architecture is implemented using PyTorch:

```python
import torch
import torch.nn as nn

class PongDQN(nn.Module):
    def __init__(self, input_channels=4, action_space=6): # input_channels is k from FrameStack
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        # Calculate the flattened size dynamically or pre-calculate based on 84x84 input
        # For input (4, 84, 84):
        # Conv1: (84-8)/4 + 1 = 19 + 1 = 20 -> (32, 20, 20)
        # Conv2: (20-4)/2 + 1 = 8 + 1 = 9   -> (64, 9, 9)
        # Conv3: (9-3)/1 + 1 = 6 + 1 = 7    -> (64, 7, 7)
        self.fc_input_dims = 7 * 7 * 64

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512), nn.ReLU(inplace=True),
            nn.Linear(512, action_space),
        )

    def forward(self, x):
        # Ensure input is float32 and on the correct device
        # x = x.float() / 255.0 # Normalization should happen in preprocess
        x = self.features(x)
        x = x.reshape(x.size(0), -1) # Use reshape for modern PyTorch, or view
        return self.fc(x)

    def _get_conv_output_shape(self, shape):
        """ Helper to compute output shape of conv layers for dynamic fc input size """
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
```
*Note: Added `input_channels` to `__init__`, used `inplace=True` for ReLUs for minor memory optimization, and used `reshape` instead of `view` which is more flexible. The `_get_conv_output_shape` is a common helper if input dimensions might vary, but here `fc_input_dims` is pre-calculated.*

## 5. Replay Buffer

A simple deque-based replay buffer is used:

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, size=int(1e5)):
        self.buffer = deque(maxlen=size)

    def store(self, transition): # transition is (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return [] # Not enough samples yet
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
```
*Note: Added a check in `sample` to prevent errors if buffer size is less than `batch_size`, and added `__len__`.*

## 6. Training Loop Structure (Conceptual)

The training loop will follow this general structure:

```python
# Conceptual - actual implementation in pong_dqn.py
# agent = DQNAgent(...)
# replay_buffer = ReplayBuffer(...)
# framestack = FrameStack(...)
# env = gym.make("ALE/PongNoFrameskip-v4", ...)

# for episode in range(max_episodes):
#     obs, info = env.reset()
#     state = framestack.reset(obs) # state is (4, 84, 84)
#     done = False
#     episode_reward = 0

#     while not done:
#         action = agent.act(state)  # ε-greedy
#         next_obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated

#         next_state = framestack.step(next_obs) # next_state is (4, 84, 84)
#         replay_buffer.store((state, action, reward, next_state, done))

#         state = next_state
#         episode_reward += reward

#         if len(replay_buffer) > agent.batch_size: # Start learning once buffer has enough samples
#             agent.learn(replay_buffer)

#     if episode % eval_interval == 0:
#         eval_reward = evaluate(agent, env, framestack) # evaluate uses agent.act with epsilon=0
#         print(f"Episode {episode}, Eval Reward: {eval_reward}, Episode Reward: {episode_reward}")
```

These patterns provide a solid foundation for the DQN implementation for Pong.
