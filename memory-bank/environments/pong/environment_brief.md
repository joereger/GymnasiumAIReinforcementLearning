# Environment Brief: Pong

**Environment Name:** `ALE/PongNoFrameskip-v4`

**Source:** OpenAI Gymnasium (via `ale-py` for Atari Learning Environment)

**Description:**
Pong is a classic two-dimensional sports game simulating table tennis. Each player controls a paddle, moving it vertically on their side of the screen. The objective is to hit the ball past the opponent's paddle. Points are scored when one player fails to return the ball to the other player. The game ends when one player reaches a predetermined score (typically 21).

**Key Characteristics for RL:**
- **Observation Space:** Pixel-based. The raw observation is an RGB image of the game screen (e.g., 210x160x3 pixels). Preprocessing (grayscaling, downsampling, frame stacking) is essential.
- **Action Space:** Discrete. Typically includes actions like NOOP (no operation), FIRE (to start a new point or serve), UP (move paddle up), DOWN (move paddle down), and possibly combinations if the specific environment variant supports them. For `PongNoFrameskip-v4`, the typical actions used are UP and DOWN, with FIRE used to start a point.
- **Reward System:**
    - +1 for scoring a point against the opponent.
    - -1 when the opponent scores a point.
    - 0 for all other time steps.
- **Episode Termination:** An episode ends when one player reaches 21 points.

**Chosen Variant:** `ALE/PongNoFrameskip-v4`
- **Reasoning:** This variant is standard for DQN research. It provides direct control over frame skipping and stacking, which will be handled by the agent's preprocessing logic.
- **Instantiation:** `env = gym.make("ALE/PongNoFrameskip-v4", repeat_action_probability=0.0)` (Note: `repeat_action_probability` is often set to 0 or a low value when manual frame skipping is implemented, but the default for Atari environments in Gymnasium might handle this. The user specifically provided 0.0).

**Objective:**
Train an agent to play Pong effectively, maximizing its score against the built-in opponent.
