# Pong Environment System Patterns

This document outlines the standard patterns and best practices to follow when working with the Pong environment in our RL implementations.

## Environment Wrappers

The Pong environment MUST be properly wrapped with the following wrappers in this specific order:

```python
def make_atari_env(env_id, render_mode=None, max_episode_steps=None):
    """Create a properly wrapped Atari environment."""
    env = gym.make(env_id, render_mode=render_mode, repeat_action_probability=0.0, full_action_space=False)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    # Apply wrappers in the standard order
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = StackFrame(env, 4)

    return env
```

### Critical Wrappers and Their Purpose

1. **NoopResetEnv**: Performs random number of no-op actions on reset to randomize initial state
2. **MaxAndSkipEnv**: Applies frame skipping (4 frames) and max pooling
3. **FireResetEnv**: ESSENTIAL for Pong - presses FIRE to start the game
4. **EpisodicLifeEnv**: Treats loss of life as episode end for better value estimation
5. **WarpFrame**: Resizes observations to 84x84 and converts to grayscale
6. **ClipRewardEnv**: Clips rewards to {-1, 0, 1} for stable learning
7. **StackFrame**: Stacks 4 frames together and converts to channels-first format

## Tensor Handling

All neural network models MUST use PyTorch's channels-first format (C, H, W) for image inputs:

```python
# Ensure state has correct format (channels first for PyTorch)
if len(state.shape) == 3 and state.shape[0] != self.state_shape[0]:
    # Convert from (H, W, C) to (C, H, W) if needed
    state = np.transpose(state, (2, 0, 1))
```

The standard state shape should be `(4, 84, 84)` for 4 stacked frames of 84x84 grayscale images.

## Reward Handling

- All rewards should be clipped to {-1, 0, 1} using the ClipRewardEnv wrapper
- For DQN implementations, rewards are already clipped by the wrapper
- For PPO implementations, we use the raw rewards from the environment after wrapper preprocessing

## RL Algorithm Implementations

We maintain implementations of multiple RL algorithms for Pong:

1. **DQN (Deep Q-Network)**:
   - Standard architecture: 3 conv layers + 2 FC layers
   - Uses experience replay buffer and target network
   - Hyperparameters: γ=0.99, ε decay over 1M frames, target updates every 10K steps

2. **Double DQN**:
   - Same architecture as DQN but with double Q-learning
   - Helps reduce overestimation bias in Q-values
   - Modified hyperparameters: lower learning rate, more frequent target updates

3. **PPO (Proximal Policy Optimization)**:
   - Actor-critic architecture with shared feature extractor
   - Uses GAE for advantage estimation
   - Clip parameter: 0.2, multiple epochs per batch update

## File Structure

Each implementation has a set of standard file components:

1. **Utilities Module** (`pong_<algorithm>_utils.py`): 
   - Contains environment wrappers
   - Replay buffer or rollout buffer implementation
   - Helper functions for data processing

2. **Model Module** (`pong_<algorithm>_model.py`):
   - Neural network architecture
   - Agent class with act and learning methods
   - Model saving/loading functionality

3. **Training Module** (`pong_<algorithm>_train.py`):
   - Training loop
   - Evaluation function
   - Hyperparameter settings

4. **Visualization Module** (`pong_<algorithm>_vis.py`):
   - Functions for plotting training progress
   - Stats recording and visualization

## Common Mistakes to Avoid

1. **Not using FireResetEnv**: Pong requires pressing FIRE to start - without this, the agent will see no movement
2. **Incorrect tensor dimensions**: Using channels-last format (H, W, C) instead of channels-first (C, H, W)
3. **Improper hyperparameters**: Each algorithm requires specific hyperparameter tuning
4. **Missing wrappers**: All wrappers are necessary for proper environment behavior
5. **Overwriting frame stacking**: The StackFrame wrapper already provides stacked frames - no need for additional stacking
