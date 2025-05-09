# Environment Brief: Bipedal Walker

**Environment Source:** OpenAI Gymnasium (`BipedalWalker-v3` or `BipedalWalkerHardcore-v3`)

**Goal:**
Train a 2D bipedal robot to walk across a randomly generated terrain. The robot has two legs, each with two joints (hip and knee).

**Observation Space:**
A 24-dimensional vector containing:
- Hull angle speed
- Angular velocity
- Horizontal speed
- Vertical speed
- Joint positions (hip1, knee1, hip2, knee2)
- Joint speeds (hip1, knee1, hip2, knee2)
- 10 lidar rangefinder measurements detecting the terrain ahead.
- Leg contact with ground (leg1_contact, leg2_contact)

**Action Space:**
A 4-dimensional continuous vector representing torques applied to the four joints:
- Hip 1 (left)
- Knee 1 (left)
- Hip 2 (right)
- Knee 2 (right)
Values are typically in the range [-1, 1].

**Reward Structure:**
- **Moving forward:** Positive reward for progressing to the right.
- **Effort:** Small negative reward for motor torque usage.
- **Falling:** Large negative reward (-100) if the walker's hull touches the ground.
- **Reaching the end:** Large positive reward (+300 for `BipedalWalker-v3`, variable for hardcore) for successfully traversing the terrain.

**Termination Conditions:**
- The walker's hull makes contact with the ground (falls over).
- The walker successfully reaches the end of the terrain.
- Episode length exceeds a maximum limit (e.g., 1600-2000 steps).

**Key Challenges:**
- High-dimensional continuous observation and action spaces.
- Balancing forward movement with stability.
- Efficiently using motor torques.
- Adapting to randomly generated, uneven terrain.
- The "hardcore" version introduces more difficult terrain features like stumps, pitfalls, and stairs.
