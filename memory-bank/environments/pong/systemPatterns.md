# System Patterns: Pong (`PongNoFrameskip-v4`)

This document details key implementation patterns and architectural decisions for the Pong environment.

## Structural Issues Identified

After implementing DQN, Double DQN, and PPO approaches with no success (all stuck at -21 reward), we identified several critical structural issues that were preventing successful learning:

1. **Missing Atari Wrappers:**
   * Our initial implementations lacked critical environment wrappers required for successful Atari learning:
     * `NoopResetEnv`: Randomizes initial states by taking random no-ops on reset
     * `MaxAndSkipEnv`: Performs frame skipping and max-pooling across frames
     * `FireResetEnv`: Pong environment requires pressing FIRE to start
     * `EpisodicLifeEnv`: Treats end-of-life as episode end (helps value estimation)

2. **Observation Processing:**
   * **Channel Order Issues**: PyTorch expects channels-first ordering (C, H, W), but our observations were channels-last (H, W, C)
   * **Normalization**: Inconsistent normalization across implementations

3. **Action Space Issues:**
   * Pong only requires 3 meaningful actions (UP, DOWN, NOOP), but we were using all 6
   * Full action space creates a larger exploration challenge

4. **Hyperparameter Tuning:**
   * Learning rates were often too high (causing instability) or too low (causing slow learning)
   * Target network update frequency was inappropriate for the environment
   * Buffer sizes and batch sizes were suboptimal

## Correct Implementation Patterns

Successful implementations should follow these patterns:

### 1. Environment Setup

```python
def make_atari_env(env_id, render_mode=None, max_episode_steps=None):
    """Create a properly wrapped Atari environment."""
    env = gym.make(env_id, render_mode=render_mode, repeat_action_probability=0.0, full_action_space=False)
    
    # Apply wrappers in specific order
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)

    return env
```

### 2. Observation Processing

* **Preprocessing Order:**
  * Convert to grayscale → Resize to 84x84 → Normalize to [0, 1] → Stack frames
  
* **Correct Input Shape for PyTorch:** 
  * Input shape should be (4, 84, 84) for 4 stacked frames (channels first)
  * Convert observations if needed:
  ```python
  # Convert from (H, W, C) to (C, H, W) if needed
  if len(state.shape) == 3 and state.shape[0] != input_shape[0]:
      state = np.transpose(state, (2, 0, 1))
  ```

### 3. Effective Hyperparameters

For DQN-based approaches:
* Learning rate: ~1e-4
* Buffer size: 100,000-500,000 transitions
* Target network update: Every 1,000-10,000 steps
* Epsilon decay: Over 250,000-1,000,000 frames

For PPO:
* Learning rate: 2.5e-4
* Clip parameter: 0.1-0.2
* GAE lambda: 0.95
* Value coefficient: 0.5
* Entropy coefficient: 0.01
* PPO epochs per update: 4
* Rollout length: 128-2048 steps

### 4. Training Patterns

* **Episode Structure:**
  * Use a fixed number of steps rather than episodes as the training metric
  * Implement proper early stopping for terminal states
  * Handle environment resets correctly (with FireReset for Pong)

* **Reward Handling:**
  * Clip rewards to {-1, 0, 1} (done by ClipRewardEnv wrapper)
  * For PPO, use proper Generalized Advantage Estimation (GAE)

* **Visualization and Debugging:**
  * Track policy entropy as a key metric (should start high and gradually decrease)
  * Monitor loss values for signs of divergence or collapse
  * Save and examine preprocessed frames to verify correct state representation

## Implementation Results

Our new implementations with these fixes:

1. **Baselines-style Double DQN:**
   * Properly implements the OpenAI Baselines approach to DQN
   * Includes all standard Atari wrappers
   * Uses correct PyTorch tensor shapes and processing
   * Includes thorough visualization and monitoring

2. **Fixed PPO:**
   * Implements all standard PPO components with proper hyperparameters
   * Includes channels-first ordering for PyTorch
   * Uses correct advantage estimation and clipped surrogate objective
   * Implements larger batch sizes and appropriate buffer sizes

The diagnostic tool helps identify preprocessing, observation shapes, and action space issues, confirming these were the root causes of learning failures in previous implementations.
