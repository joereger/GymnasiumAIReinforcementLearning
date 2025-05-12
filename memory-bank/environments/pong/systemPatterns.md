# Pong Environment System Patterns

This document outlines the standard patterns and best practices to follow when working with the Pong environment in our RL implementations.

## Action Space

**CRITICAL UPDATE**: We now use a reduced action space of only 3 actions for Pong:

```python
# Original actions: NOOP(0), FIRE(1), RIGHT(2), LEFT(3), RIGHTFIRE(4), LEFTFIRE(5)
# For Pong:
# - RIGHT (2) moves paddle UP
# - LEFT (3) moves paddle DOWN
# - NOOP (0) keeps paddle in place
valid_actions = [0, 2, 3]  # [STAY, UP, DOWN]
```

This simplifies learning significantly as the agent only needs to control vertical paddle movement. The FIRE action is only needed during reset to start the game (handled by the FireResetEnv wrapper).

## Environment Wrappers

The Pong environment MUST be properly wrapped with the following wrappers in this specific order:

```python
def make_atari_env(env_id, render_mode=None, max_episode_steps=None, reduced_actions=True):
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
    
    # Reduce action space to simplify learning (default True)
    if reduced_actions:
        env = ReducedActionSpace(env)

    return env
```

### Critical Wrappers and Their Purpose

1.  **NoopResetEnv**: Performs random number of no-op actions on reset to randomize initial state.
2.  **MaxAndSkipEnv**: Applies frame skipping (4 frames) and max pooling over the last 2 frames.
3.  **FireResetEnv**: ESSENTIAL for Pong - presses FIRE to start the game.
4.  **EpisodicLifeEnv**: Treats loss of life as episode end for better value estimation.
5.  **WarpFrame**: Resizes observations to 84x84 and converts to grayscale.
6.  **ClipRewardEnv**: Clips rewards to {-1, 0, 1} for stable learning.
7.  **StackFrame**: Stacks 4 frames together and converts to channels-first format `(4, 84, 84)` normalized to `[0, 1]`.
8.  **ReducedActionSpace**: Reduces action space to 3 essential actions (STAY, UP, DOWN).

## Tensor Handling

All neural network models MUST use PyTorch's channels-first format `(4, 84, 84)` for image inputs. The `StackFrame` wrapper handles this.

## DQN Hyperparameters (Simplified Baseline)

For a barebones DQN implementation aimed at verifying basic learning:

*   `LEARNING_RATE = 2.5e-4`
*   `BATCH_SIZE = 32`
*   `REPLAY_BUFFER_SIZE = int(5e4)` (50,000)
*   `GAMMA = 0.99`
*   `TARGET_UPDATE_FREQ = 1000` (frames)
*   `EPSILON_START = 1.0`
*   `EPSILON_END = 0.01`
*   `EPSILON_DECAY_FRAMES = int(1e5)` (100,000)
*   `MAX_FRAMES_TOTAL = int(5e5)` (500,000 for quicker tests)

**No Replay Buffer Warmup**: Learning begins as soon as the replay buffer contains `BATCH_SIZE` experiences.

## Evaluation Considerations

When evaluating trained models:

1.  Use `explore=False` in `agent.act()` for a greedy policy, but be aware this can perform poorly if the Q-values are not well-learned.
2.  Consider using a small `eval_epsilon` (e.g., 0.05) during evaluation to allow some exploration, which can give a more robust measure of performance, especially for partially trained agents.
3.  Ensure the evaluation environment uses the same wrappers as the training environment, including `ReducedActionSpace`.

## Common Mistakes to Avoid

1.  **Not using FireResetEnv**: Pong requires pressing FIRE to start.
2.  **Incorrect tensor dimensions**: Ensure PyTorch models receive `(C, H, W)` format.
3.  **Using all 6 actions**: The reduced 3-action space is crucial for efficient learning.
4.  **Missing wrappers**: All listed wrappers play a role.
