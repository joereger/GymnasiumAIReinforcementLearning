# Environment Brief: Cart Pole

**Environment Source:** OpenAI Gymnasium (`CartPole-v1`)

**Goal:**
Balance a pole upright on a cart that moves along a frictionless track. The system is controlled by applying a horizontal force to the cart.

**Observation Space:**
A 4-dimensional vector containing:
- Cart Position (meters)
- Cart Velocity (m/s)
- Pole Angle (radians, 0 is upright)
- Pole Angular Velocity (rad/s)

**Action Space:**
A discrete action space with 2 actions:
- 0: Push cart to the left
- 1: Push cart to the right

**Reward Structure:**
- A reward of +1 is provided for every timestep that the pole remains upright.
- The episode ends if the pole angle exceeds ±12 degrees (0.209 radians) from vertical, or if the cart moves more than ±2.4 units from the center.
- The maximum episode length is typically 500 steps for `CartPole-v1`. An unsolved environment might have a limit of 200 steps.

**Termination Conditions:**
- Pole angle goes beyond ±12 degrees (0.209 radians) from vertical.
- Cart position moves more than ±2.4 units from the center of the track.
- Episode length reaches the maximum limit (e.g., 500 for v1).

**Key Challenges:**
- Classic control problem, often used as a "hello world" for reinforcement learning.
- Relatively simple dynamics but requires learning a stable control policy.
- Balancing exploration (trying new actions) with exploitation (using known good actions).
- The system is inherently unstable; without control, the pole will quickly fall.
