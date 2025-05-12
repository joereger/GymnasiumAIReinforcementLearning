# Pong Environment Brief

## Overview

Pong is one of the classic Atari games in the OpenAI Gymnasium environment suite. It simulates a simple table tennis game where two paddles hit a ball back and forth. The agent controls one paddle and aims to defeat the built-in AI opponent.

## Environment Details

- **ID**: `PongNoFrameskip-v4`
- **Observation Space**: RGB image (210, 160, 3)
  - After processing: Grayscale images stacked in proper channels-first format (4, 84, 84)
- **Action Space**: Discrete(6)
  - Action 0: NOOP
  - Action 1: FIRE (starts the game)
  - Action 2: RIGHT (move paddle up in Pong)
  - Action 3: LEFT (move paddle down in Pong)
  - Action 4: RIGHTFIRE 
  - Action 5: LEFTFIRE
- **Reward**: +1 when the agent scores, -1 when the opponent scores

## Key Challenges

1. **Visual Input Preprocessing**: Raw pixel inputs must be properly processed:
   - Convert to grayscale
   - Resize to 84x84
   - Normalize pixel values
   - Stack 4 frames together
   - Ensure proper channels-first ordering for PyTorch

2. **Sparse Rewards**: Rewards are very sparse in Pong (only when a point is scored)

3. **Delayed Feedback**: There's a delay between actions and their outcomes

4. **Essential Environment Wrappers**:
   - `NoopResetEnv`: Start each episode with random number of no-ops
   - `MaxAndSkipEnv`: Skip frames and perform max-pooling over skipped frames
   - `FireResetEnv`: Press FIRE to start the game
   - `EpisodicLifeEnv`: End episode when a life is lost
   - `ClipRewardEnv`: Clip rewards to {-1, 0, 1}

## Successful Approach Requirements

The most effective approaches for Pong combine:

1. **Proper Frame Processing**:
   - Correct channels ordering (PyTorch uses channels-first format)
   - Frame stacking for temporal information

2. **Effective Exploration**:
   - Epsilon-greedy with appropriate decay schedule
   - Starts highly exploratory and gradually becomes more exploitative

3. **Stable Learning**:
   - Target networks updated periodically
   - Large replay buffer (100k-1M transitions)
   - Gradient clipping to prevent exploding gradients

4. **Critical Environment Wrappers**:
   - Especially FireResetEnv (otherwise game doesn't start)
   - Clipping rewards to stabilize learning
