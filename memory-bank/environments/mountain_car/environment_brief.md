# Environment Brief: Mountain Car

**Environment Source:** OpenAI Gymnasium (`MountainCar-v0`, `MountainCarContinuous-v0`)

**Goal:**
Drive an underpowered car up a steep hill to reach a flag located at the top (position 0.5). The car's engine is not strong enough to drive directly up the hill, so it must build momentum by driving back and forth.

**Observation Space:**
A 2-dimensional vector:
- Position of the car (between -1.2 and 0.6)
- Velocity of the car (between -0.07 and 0.07)

**Action Space:**
- **Discrete (`MountainCar-v0`):** 3 actions
    - 0: Accelerate to the left
    - 1: Do nothing (coast)
    - 2: Accelerate to the right
- **Continuous (`MountainCarContinuous-v0`):** 1-dimensional vector
    - Horizontal force to apply to the car (between -1.0 and 1.0).

**Reward Structure:**
- **`MountainCar-v0` (Discrete):**
    - -1 for each time step until the goal is reached.
    - 0 upon reaching the goal.
- **`MountainCarContinuous-v0`:**
    - +100 for reaching the goal state (position 0.5).
    - Minus the square of the action magnitude (e.g., `action[0]*action[0]*0.1`). This penalizes large actions.

**Termination Conditions:**
- The car reaches the goal position (position >= 0.5).
- Episode length exceeds a maximum limit (e.g., 200 steps for `MountainCar-v0`, 999 for `MountainCarContinuous-v0`).

**Key Challenges:**
- Sparse rewards: In the discrete version, the agent only receives a non-negative reward upon reaching the goal, making exploration difficult.
- Credit assignment: Actions taken early on are crucial for building momentum, but their effect is delayed.
- Continuous version requires fine control of acceleration.
- Learning the strategy of oscillating to build momentum.
