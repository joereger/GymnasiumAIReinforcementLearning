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
def make_pong_env(env_id="PongNoFrameskip-v4", render_mode=None, reduced_actions=True, seed=None):
    """Create a preprocessed Pong environment with all necessary wrappers."""
    env = gym.make(env_id, render_mode=render_mode, repeat_action_probability=0.0, full_action_space=False)
    
    if seed is not None:
        env.seed(seed)
        
    # Apply wrappers in the standard order
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = ChannelsFirstImageShape(env)  # PyTorch uses CHW format
    env = ScaledFloatFrame(env)
    
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
7.  **FrameStack**: Stacks 4 frames together.
8.  **ChannelsFirstImageShape**: Converts to channels-first format (4, 84, 84).
9.  **ScaledFloatFrame**: Normalizes pixel values to [0, 1].
10. **ReducedActionSpace**: Reduces action space to 3 essential actions (STAY, UP, DOWN).

## Tensor Handling

All neural network models MUST use PyTorch's channels-first format `(4, 84, 84)` for image inputs. The `ChannelsFirstImageShape` wrapper handles this conversion.

## PPO Implementation (Current Approach)

We are currently using PPO (Proximal Policy Optimization) as our primary algorithm, which has shown better stability and sample efficiency than DQN for this environment.

### PPO Hyperparameters

Updated PPO hyperparameters for Pong (as of 5/11/2025):

```python
# PPO hyperparameters
learning_rate = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
value_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5
rollout_steps = 128
n_epochs = 4
batch_size = 64
total_timesteps = 25_000_000  # Increased to 25M for thorough learning
eval_freq = 500  # Evaluate every 500 episodes
save_freq = 100  # Save model every 100 episodes
```

### Key PPO Components

1. **Actor-Critic Architecture**: Using a shared CNN backbone with separate policy (actor) and value (critic) heads.
2. **Generalized Advantage Estimation (GAE)**: For more stable advantage calculation.
3. **PPO Clipping**: Restricts policy updates to prevent destructively large updates.
4. **Entropy Regularization**: Encourages exploration by penalizing overly deterministic policies.
5. **Multiple Epochs**: Running several optimization passes per rollout for better sample efficiency.
6. **Best Model Tracking**: Saves a separate copy of the best-performing model during evaluation.
7. **Training Continuity**: Supports stopping and resuming training with automatic progress recovery.

### Rollout Collection

PPO collects experience in rollouts (typically 128 steps) rather than using a replay buffer:

```python
# Collect rollout
rollout = collect_rollout(
    env=env,
    agent=agent,
    ppo=ppo,
    rollout_steps=args.rollout_steps,
    render=train_render  # Optional rendering during training
)

# Update agent with multiple optimization epochs
loss_metrics = ppo.update(
    rollout=rollout,
    n_epochs=4,
    batch_size=64
)
```

### Training Data Persistence

Training progress is saved in a structured JSON format that facilitates resuming training:

```python
# Each episode's data is saved as an object with all metrics
episode_data.append({
    'episode': episodes,
    'reward': current_episode_reward,
    'length': current_episode_length,
    'time': episode_duration,
    'policy_loss': avg_policy_loss,
    'value_loss': avg_value_loss,
    'entropy': avg_entropy,
    'cumulative_time': cumulative_time,
    'timesteps': total_timesteps,
    'eval_reward': eval_reward  # Only present after evaluation
})
```

When loading a model, the code automatically looks for and loads this progress data to continue training seamlessly.

## DQN Hyperparameters (Alternative Approach)

For DQN implementations (now a secondary approach):

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

1. **Deterministic vs Stochastic Policy**:
   - For PPO, use `deterministic=True` in `agent.get_action()` for evaluation.
   - For DQN, use `explore=False` in `agent.act()` for a greedy policy.

2. **Limited Exploration**: We use a small exploration rate (epsilon=0.05) during evaluation to get a more realistic measure of performance, especially for partially trained agents.

3. **Updated Evaluation Protocol**:
   - Evaluate every 500 episodes (rather than by steps)
   - Run 5 complete episodes per evaluation
   - Record average reward and episode length
   - Save evaluation results in the training progress JSON
   - Track and save the best-performing model separately

4. **Consistent Environment**: The evaluation environment uses the same wrappers as the training environment, including `ReducedActionSpace`, but with `render_mode=None` by default.

5. **Visualization**:
   - Use a single chart with multiple y-axes for clear visualization
   - Primary axis for rewards, secondary axis for losses and entropy
   - Include proper legends and clear color coding

## Common Mistakes to Avoid

1. **Not using FireResetEnv**: Pong requires pressing FIRE to start.
2. **Incorrect tensor dimensions**: Ensure PyTorch models receive `(C, H, W)` format.
3. **Using all 6 actions**: The reduced 3-action space is crucial for efficient learning.
4. **Missing wrappers**: All listed wrappers play a role.
5. **PPO-specific**: Not normalizing advantages can lead to unstable learning.
6. **PPO-specific**: Using too short rollouts reduces learning signal quality.
7. **Missing sound disabling**: Game sound can cause unnecessary distraction during training.
8. **Not handling training continuity**: Failing to reload training progress data when continuing training.
9. **Forgetting to track best model**: Always keep a separate copy of the best-performing model.
10. **Using improper seeding**: Modern Gymnasium uses `env.reset(seed=...)` rather than `env.seed()`.
