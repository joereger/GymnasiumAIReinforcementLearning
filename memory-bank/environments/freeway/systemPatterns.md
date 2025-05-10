# System Patterns: Freeway

This document outlines the system architecture, patterns, and key technical decisions in the Freeway implementation.

## Overall Architecture

The Freeway implementation follows a modular architecture with clear separation of concerns:

```
┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Environment    │────►│  Agent Components  │────►│  Training Loop  │
│  Wrappers       │     │  & Neural Network  │     │  & Evaluation   │
└─────────────────┘     └───────────────────┘     └─────────────────┘
```

### Key Components

1. **Environment Wrappers**:
   - `PreprocessFreeway` - Image preprocessing pipeline
   - `CustomRewardFreeway` - Enhanced reward engineering

2. **Agent Components**:
   - `DQNNetwork` - Neural network architecture
   - `ReplayBuffer` - Experience storage
   - `DQNAgent` - Core agent logic

3. **Training & Evaluation**:
   - `train_agent` - Training loop
   - `evaluate_agent` - Evaluation function
   - `plot_training_progress` - Visualization utility

## Design Patterns

### 1. Wrapper Pattern

The implementation uses the Gymnasium Wrapper pattern extensively:

```python
class PreprocessFreeway(gym.Wrapper):
    def __init__(self, env, frame_stack=4, resize_shape=(84, 84)):
        super(PreprocessFreeway, self).__init__(env)
        # ...
    
    def reset(self, **kwargs):
        # Process the reset from wrapped environment
        # ...
    
    def step(self, action):
        # Process the step from wrapped environment
        # ...
```

This pattern allows:
- Stacking multiple processing steps
- Maintaining the standard Gymnasium interface
- Clear separation of preprocessing and reward modification

### 2. Buffer Pattern

The replay buffer implements a circular buffer pattern:

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
```

Key aspects:
- Efficient memory usage with circular overwriting
- Random sampling for decorrelation
- Fixed capacity to prevent memory issues

### 3. Target Network Pattern

The implementation uses the target network pattern from DQN:

```python
# Initialize networks
self.policy_net = DQNNetwork(state_shape, n_actions).to(device)
self.target_net = DQNNetwork(state_shape, n_actions).to(device)
self.target_net.load_state_dict(self.policy_net.state_dict())
self.target_net.eval()

# Periodically update target network
if self.train_steps % self.target_update_freq == 0:
    self.target_net.load_state_dict(self.policy_net.state_dict())
```

This pattern:
- Stabilizes learning by providing fixed targets
- Prevents oscillations and divergence during training
- Implements "soft targets" approach from original DQN paper

### 4. Checkpoint/Resume Pattern

The implementation allows saving and loading training progress:

```python
def save(self, filename):
    torch.save({
        'policy_net': self.policy_net.state_dict(),
        'target_net': self.target_net.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'epsilon': self.epsilon,
        'train_steps': self.train_steps,
    }, filename)

def load(self, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        # ...
```

This enables:
- Resuming training from previous sessions
- Saving best-performing models
- Evaluation of trained models

## Critical Implementation Paths

### 1. Preprocessing Pipeline

```
Raw Observation → Grayscale → Resize → Frame Stacking → Normalization
```

This path is critical as it determines the input representation for the neural network.

### 2. Reward Engineering

```
Original Reward → Zone Progress Bonus → Movement Reward/Penalty → Time Penalty → Final Reward
```

This path shapes the learning process and addresses the sparse reward challenge.

### 3. Action Selection Path

```
State → ε-greedy Decision → Policy Network Inference → Action Selection
```

This path determines the agent's behavior during training and evaluation.

### 4. Learning Update Path

```
Experience Storage → Batch Sampling → Target Computation → Loss Calculation → Network Update
```

This path implements the core DQN algorithm and drives the learning process.

## Key Technical Decisions

### 1. Reward Engineering Approach

**Decision**: Implement multiple complementary reward signals instead of a single modified reward.

**Rationale**: 
- Sparse rewards in Freeway make learning difficult
- Different reward components address different aspects of desired behavior
- Provides smoother learning gradient across the state space

### 2. Frame Stack Size

**Decision**: Stack 4 frames rather than using a different number.

**Rationale**:
- Standard in DQN literature
- Balances temporal information with memory efficiency
- Captures sufficient motion information for car avoidance

### 3. Network Architecture

**Decision**: Use 4 convolutional layers and 3 fully connected layers.

**Rationale**:
- Deeper than original DQN (which used 3 conv layers)
- Additional layers provide more representational capacity
- Extra layer helps capture complex visual patterns in traffic

### 4. RAM-Based Position Tracking with Robust Error Handling

**Decision**: Use direct RAM access for position tracking with comprehensive error handling and debugging.

**Rationale**:
- More reliable and precise position information
- Significantly faster than computer vision approaches
- Enables accurate reward shaping based on position
- Graceful degradation if memory layout varies
- Self-diagnosing capabilities during initialization

### 5. Training Visualization

**Decision**: Allow optional rendering during training with toggle.

**Rationale**:
- Visual feedback helps understand agent behavior
- Optional to allow faster training when not needed
- Useful for debugging reward engineering effects
