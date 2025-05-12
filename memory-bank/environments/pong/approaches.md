# Pong Implementation Approaches

This document details the various approaches we've implemented for the Pong environment and lessons learned from each.

## Approach 1: Standard DQN

Our initial DQN implementation follows the architecture from the 2015 DeepMind Atari paper but initially had structural issues that prevented successful learning.

**Key Components:**
- **Architecture**: 3 convolutional layers followed by 2 fully connected layers
- **Input**: 4 stacked frames (84x84 grayscale)
- **Learning Rate**: 2.5e-4
- **Exploration**: Epsilon-greedy with exponential decay
- **Target Network Update**: Every 10,000 steps
- **Replay Buffer**: 100,000 samples
- **Optimizer**: Adam

**Performance before fixes**:
- Unable to improve beyond the random policy (-21 reward)
- Unable to score any points consistently

**Critical Issues Identified:**
1. Missing essential environment wrappers (FireResetEnv, NoopResetEnv, etc.)
2. Incorrect tensor dimension ordering (using channels-last instead of channels-first)
3. No proper handling of Pong-specific requirements (pressing FIRE to start)

## Approach 2: Fixed DQN Implementation

We made the following critical fixes to our DQN implementation:

1. **Proper Environment Wrappers**:
   ```python
   def make_atari_env(env_id, render_mode=None, max_episode_steps=None):
       env = gym.make(env_id, render_mode=render_mode, repeat_action_probability=0.0, full_action_space=False)
       # Apply wrappers in specific order
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

2. **Channels-First Tensor Handling**:
   ```python
   # Ensure state has correct format (channels first for PyTorch)
   if len(state.shape) == 3 and state.shape[0] != self.state_shape[0]:
       # Convert from (H, W, C) to (C, H, W) if needed
       state = np.transpose(state, (2, 0, 1))
   ```

3. **Hyperparameter Tuning**:
   - Using learning rate of 1e-4 to 2.5e-4
   - Employing smoother epsilon decay (linear over 100k frames)
   - Target network updates every 1000 steps
   - Gradient clipping to stabilize learning

**Performance after fixes**:
- Successfully improves beyond random policy
- Able to score points and shows continuous improvement

## Approach 3: PPO Implementation

We implemented Proximal Policy Optimization (PPO) as an alternative approach, which often performs better with fewer samples than DQN.

**Key Components:**
- **Architecture**: Actor-Critic network with shared feature extractor
- **Learning Rate**: 2.5e-4
- **Discount Factor (γ)**: 0.99
- **GAE Lambda**: 0.95
- **Clip Parameter**: 0.1
- **Value Loss Coefficient**: 0.5
- **Entropy Coefficient**: 0.01
- **Rollout Length**: 2048 steps
- **Mini-batch Size**: 64

**Critical Implementation Details:**
1. **Advantage Calculation**:
   - Using Generalized Advantage Estimation (GAE)
   ```python
   # Compute GAE
   gae = 0
   for t in reversed(range(len(rewards))):
       delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
       gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
       advantages[t] = gae
   ```

2. **PPO Clip Objective**:
   ```python
   # Ratio between new and old policies
   ratio = torch.exp(new_log_probs - batch_old_log_probs)
   
   # Clipped surrogate objective
   surr1 = ratio * batch_advantages
   surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
   policy_loss = -torch.min(surr1, surr2).mean()
   ```

3. **Multiple Epochs per Update**:
   - Performing 4 epochs of optimization per rollout
   - This helps extract more learning signal from each set of samples

## Results Comparison

| Approach | Training Speed | Sample Efficiency | Final Performance |
|----------|----------------|-------------------|-------------------|
| Original DQN | Slow | Poor | Unable to learn (stuck at -21) |
| Fixed DQN | Medium | Moderate | Good (reaches positive scores) |
| PPO | Fast | Good | Very Good (learns more efficiently) |

## Key Lessons

1. **Essential Environment Wrappers**: 
   - **FireResetEnv**: Critical for Pong - without this, the game never starts properly
   - **NoopResetEnv**: Important for proper exploration with random starting states
   - **EpisodicLifeEnv**: Helps with value estimation by ending episodes at life loss

2. **Tensor Dimensions Matter**: 
   - PyTorch uses channels-first format (C, H, W)
   - Incorrect channel ordering completely prevents feature learning

3. **Reward Clipping**:
   - Clipping rewards to {-1, 0, 1} significantly stabilizes learning
   - Critical for both DQN and PPO approaches

4. **Preprocessing Order**:
   - The order of operations matters: grayscale → resize → normalize → stack
   - Modifying this order can lead to information loss or distortion

5. **Diagnostic Tools Value**:
   - The diagnostic script helped identify issues quickly
   - Visualizing preprocessed frames and action distributions is critical for debugging
