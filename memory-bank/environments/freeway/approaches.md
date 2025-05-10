# Approaches: Freeway

This document details the approaches implemented for the Freeway environment.

## Approach 1: Deep Q-Network (DQN) with Custom Rewards

**Status**: Initial implementation, baseline approach

## Approach 2: Double DQN with Frame Skipping
 
- **Source File:** `freeway/freeway_double_dqn.py`
- **Algorithm:** Double Deep Q-Network with frame skipping, experience replay, and target network

### Implementation Overview

This implementation uses a more sophisticated Double DQN architecture with standard Atari preprocessing techniques to address the limitations of the basic DQN approach.

### Key Technical Components

1. **Double DQN Algorithm**:
   - Uses separate networks for action selection and evaluation
   - Reduces overestimation bias in vanilla DQN
   - More stable learning and better convergence

2. **Frame Skipping**:
   - Each action repeats for 4 frames
   - Uses max pooling over last 2 frames to reduce flickering
   - Significantly speeds up training (4x fewer decisions needed)

3. **Standard Network Architecture**:
   - 3-layer CNN architecture (vs 4 layers in Approach 1)
   - Based on the proven architecture from DeepMind's Atari papers
   - More efficient feature extraction

### Neural Network Architecture

```
Input: 4×84×84 (4 stacked grayscale frames)

Convolutional Layers:
1. Conv2D: 32 filters, 8×8 kernel, stride 4, ReLU
2. Conv2D: 64 filters, 4×4 kernel, stride 2, ReLU
3. Conv2D: 64 filters, 3×3 kernel, stride 1, ReLU

Fully Connected Layers:
1. Dense: 512 units, ReLU
2. Dense: 3 units (Q-values for NOOP, UP, DOWN actions)
```

### Optimized Hyperparameters

- **Learning Rate**: 0.00025 (4x smaller than Approach 1)
- **Discount Factor (γ)**: 0.99
- **Replay Buffer Size**: 100,000 (10x larger than Approach 1)
- **Batch Size**: 32
- **Target Network Update Frequency**: Every 1,000 steps
- **Exploration Strategy**: Linear ε decay from 1.0 to 0.01 over 500,000 frames
- **Frame Skip**: 4 (standard in Atari literature)

### Hardware Acceleration

- **Apple Silicon Support**: Uses PyTorch's MPS backend to accelerate training on Apple M1 hardware
- **Dynamic Device Selection**: Automatically detects and uses the most efficient compute device (MPS, CUDA, or CPU)
- **Optimized Memory Usage**: Better suited for GPU acceleration with simplified network architecture

### Performance Advantages

- Theoretically capable of achieving much higher scores (30+ points) with sufficient training
- More sample-efficient due to Double DQN algorithm
- Faster training due to frame skipping and GPU acceleration
- Natural exploration without action biasing leads to better policy learning

### Results & Performance

(To be filled in after training runs)

### Learnings & Insights

(To be filled in after analyzing training results)

- **Source File:** `freeway/freeway.py`
- **Algorithm:** Deep Q-Network (DQN) with experience replay and target network

### Implementation Overview

The implementation uses a Deep Q-Network architecture with several enhancements to deal with the challenges of the Freeway environment, particularly the sparse reward structure.

### Preprocessing Pipeline

```
Raw RGB Image (210×160×3) → Grayscale → Resize (84×84) → Stack 4 frames → Normalize [0,1]
```

- **Grayscale Conversion**: Reduces dimensionality while preserving essential features
- **Resizing**: Standard 84×84 format used in DQN literature
- **Frame Stacking**: Captures motion/temporal information by stacking 4 consecutive frames
- **Normalization**: Scales pixel values to [0,1] range for stable training

### Neural Network Architecture

```
Input: 4×84×84 (4 stacked grayscale frames)

Convolutional Layers:
1. Conv2D: 32 filters, 8×8 kernel, stride 4, ReLU
2. Conv2D: 64 filters, 4×4 kernel, stride 2, ReLU
3. Conv2D: 64 filters, 3×3 kernel, stride 1, ReLU
4. Conv2D: 32 filters, 3×3 kernel, stride 1, ReLU

Fully Connected Layers:
1. Dense: 512 units, ReLU
2. Dense: 256 units, ReLU
3. Dense: 3 units (Q-values for NOOP, UP, DOWN actions)
```

### Reward Engineering

Custom reward wrappers enhance the sparse rewards:

1. **Zone-Based Progress Rewards**:
   - Screen divided into 10 vertical zones
   - +0.1 reward for entering a higher zone
   - Provides intermediate feedback for upward progress

2. **Movement-Based Rewards**:
   - +0.1 * progress_fraction for upward movement
   - -0.1 * regress_fraction for downward movement
   - Directly reinforces desired behaviors

3. **Time Penalty**:
   - -0.01 per timestep
   - Encourages efficient paths and discourages stalling

4. **Original Reward**:
   - +1.0 for successfully crossing (preserved from original environment)

The combined reward function creates a smoother reward landscape that guides learning toward successful strategies while maintaining the original environment objectives.

### RAM-Based Position Tracking

A key technical component of the reward engineering system is the player position tracking mechanism:

- **RAM Address Identification**: Uses address 0x59 (89 in decimal) to track the chicken's vertical position
- **Robust Error Handling**: 
  - Includes fallback mechanisms if the primary address is inaccessible
  - Provides diagnostic RAM information during initialization (RAM size and values)
  - Safely accesses memory within array bounds
  - Gracefully handles potential memory access errors
- **Position Normalization**: Translates raw RAM values to normalized screen coordinates for zone calculations

This approach provides more reliable position tracking than pixel-based methods, enabling precise reward shaping while handling the specific memory layout of the Atari Freeway implementation.

### Key Hyperparameters

- **Learning Rate**: 1e-4
- **Discount Factor (γ)**: 0.99
- **Replay Buffer Size**: 10,000 experiences
- **Batch Size**: 32
- **Target Network Update Frequency**: Every 1,000 steps
- **Exploration Strategy**: ε-greedy with:
  - Initial ε: 1.0
  - Final ε: 0.01
  - Decay Rate: 0.995 per episode

### Training Process

1. **Experience Collection**:
   - Execute actions using ε-greedy policy
   - Store transitions (state, action, reward, next_state, done) in replay buffer

2. **Network Training**:
   - Sample mini-batches from replay buffer
   - Compute target Q-values: r + γ * max(Q(s', a'))
   - Update policy network to minimize squared TD error
   - Periodically update target network

3. **Exploration-Exploitation Balance**:
   - Gradually decay exploration rate from 1.0 to 0.01
   - Allows extensive exploration initially, transitioning to exploitation

4. **Model Persistence**:
   - Save best-performing model based on cumulative reward
   - Periodic checkpoints every 100 episodes
   - All models saved to environment-specific data directory (`data/freeway/`)

### Performance Monitoring

- **Single-Line Episode Logging**:
  ```
  Episode: 100 | Score: 3.75 | Steps: 2145 | Epsilon: 0.50 | Loss: 0.024 | Max Q: 8.76 | Memory: 8500 | Time: 0:10:30
  ```

- **Training Visualization**:
  - Option for real-time rendering during training
  - Periodic plots of rewards, episode lengths, and max Q-values

- **Evaluation Mode**:
  - Separate evaluation function with exploration disabled
  - Reports average performance across multiple episodes

### Results & Performance

(To be filled in after training runs)

### Learnings & Insights

(To be filled in after analyzing training results)

## Potential Future Approaches

1. **Rainbow DQN**: Implement multiple DQN improvements (Prioritized Experience Replay, Dueling Networks, etc.)

2. **Proximal Policy Optimization (PPO)**: Use a policy gradient method for potentially more stable learning

3. **Curiosity-Driven Exploration**: Add intrinsic rewards for visiting new states to improve exploration

4. **Curriculum Learning**: Start with easier scenarios (slower traffic) and gradually increase difficulty
