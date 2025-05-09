# Environment Brief: Lunar Lander

**Environment Source:** OpenAI Gymnasium (`LunarLander-v2`, `LunarLanderContinuous-v2`)

**Goal:**
Navigate a lander to a landing pad located at coordinates (0,0). The lander needs to come to rest on the pad with zero speed.

**Observation Space:**
An 8-dimensional vector containing:
- Lander's (x, y) coordinates.
- Lander's (vx, vy) linear velocities.
- Lander's angle.
- Lander's angular velocity.
- Two booleans indicating whether each leg is in contact with the ground.

**Action Space:**
- **Discrete (`LunarLander-v2`):** 4 actions
    - 0: Do nothing
    - 1: Fire left orientation engine
    - 2: Fire main engine
    - 3: Fire right orientation engine
- **Continuous (`LunarLanderContinuous-v2`):** 2-dimensional vector
    - Main engine thrust (0.0 to 1.0, or -1.0 to 1.0 if negative thrust is off).
    - Left/Right engine thrust (-1.0 to 1.0).

**Reward Structure:**
- **Moving towards landing pad:** Reward for moving from the top of the screen to the landing pad and zero speed.
- **Crashing:** -100 points.
- **Coming to rest:** +100 points.
- **Each leg contact:** +10 points for each leg that touches the ground.
- **Firing main engine:** -0.3 points per frame.
- **Firing side engine:** -0.03 points per frame.
Solved is 200 points.

**Termination Conditions:**
- Lander crashes (body contacts ground or moves outside window).
- Lander comes to rest on the landing pad.
- Episode length exceeds a maximum limit (e.g., 1000 steps).

**Key Challenges:**
- Balancing thrust control for navigation and soft landing.
- Managing fuel consumption (implicit through negative rewards for firing engines).
- Precise maneuvering required to hit the small landing pad with zero speed.
- Continuous version adds complexity of fine-grained thrust control.
