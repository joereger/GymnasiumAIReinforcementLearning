# Approaches: Pong (`ALE/PongNoFrameskip-v4`)

## Approach 1: Deep Q-Network (DQN) with "Smart Defaults"

**Algorithm:** Deep Q-Network (DQN)

**Rationale:** DQN is a standard and effective algorithm for pixel-based Atari environments like Pong. The "Smart Defaults" provided by the user offer a strong, battle-tested starting point.

**Source File(s):**
- `pong/pong_dqn.py` (to be created)
- `pong/visualization_utils.py` (to be created/copied)

**Key Components & Configuration (Based on "Smart Defaults"):**

1.  **Environment Choice:**
    *   **Name:** `ALE/PongNoFrameskip-v4`
    *   **Instantiation:** `env = gym.make("ALE/PongNoFrameskip-v4", repeat_action_probability=0.0)`
    *   **Reasoning:** Canonical choice for Atari DQN research, offers precise frame-skipping control via the agent.

2.  **Preprocessing Strategy:**
    *   **Grayscale Conversion:** Convert RGB frame to grayscale using luminance.
        ```python
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        ```
    *   **Downsampling:** Resize frame to 84x84 pixels.
        ```python
        # frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        ```
    *   **Normalization:** Normalize pixel values to [0, 1] (float32).
        ```python
        # frame = frame.astype(np.float32) / 255.0
        ```
    *   **Frame Stacking:** Stack the last 4 preprocessed frames to provide motion context.
        *   A `FrameStack` class will be used (see `systemPatterns.md`).

3.  **CNN Model Structure (DeepMind 2015 Atari Paper):**
    *   **Input:** (4, 84, 84) - Stacked preprocessed frames.
    *   **Architecture:**
        1.  Conv1: 32 filters, 8x8 kernel, stride 4, ReLU activation. Output: (32, 20, 20)
        2.  Conv2: 64 filters, 4x4 kernel, stride 2, ReLU activation. Output: (64, 9, 9)
        3.  Conv3: 64 filters, 3x3 kernel, stride 1, ReLU activation. Output: (64, 7, 7)
        4.  Flatten
        5.  Dense: 512 units, ReLU activation.
        6.  Output: Linear layer with `action_space` units (typically 6 for Pong: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE; or simpler if environment action space is reduced).
    *   The `PongDQN` PyTorch module will be used (see `systemPatterns.md`).

4.  **Hyperparameters (Recommended Starting Point):**
    *   **Optimizer:** Adam
    *   **Learning Rate:** `2.5e-4`
    *   **Batch Size:** `32`
    *   **Replay Buffer Size:** `1e5` (stores `(state, action, reward, next_state, done)` tuples)
    *   **Discount Factor (γ):** `0.99`
    *   **Target Network Update Frequency:** Every `1000` training steps (agent learning steps).
    *   **Exploration (ε-greedy):**
        *   Initial ε: `1.0`
        *   Final ε: `0.01`
        *   Decay Period: Over `1,000,000` frames (agent steps in environment).

5.  **DQN Agent Components:**
    *   **Q-Network (Online Network):** The `PongDQN` model described above.
    *   **Target Network:** A separate instance of `PongDQN`, with weights periodically copied from the online network.
    *   **Replay Buffer:** A deque-based buffer of size `1e5` (see `systemPatterns.md`).
    *   **Loss Function:** Mean Squared Error (MSE) or Huber Loss between predicted Q-values and target Q-values.

6.  **Training Loop Structure:**
    *   Iterate for `max_episodes`.
    *   Inside each episode:
        *   Reset environment and frame stack.
        *   Loop until `done`:
            *   Agent selects action using ε-greedy policy based on current `state`.
            *   Take step in environment: `next_obs, reward, terminated, truncated, info`.
            *   Preprocess `next_obs` and update frame stack to get `next_state`.
            *   Store `(state, action, reward, next_state, done)` in replay buffer.
            *   Update `state = next_state`.
            *   Increment `episode_reward`.
            *   Periodically call `agent.learn()`:
                *   Sample minibatch from replay buffer.
                *   Calculate target Q-values using the target network.
                *   Perform gradient descent step on the online network.
                *   Periodically update the target network.
    *   Periodically evaluate agent performance with ε=0.

**Expected Outcome:**
The agent should learn to play Pong effectively, achieving a positive average score over a series of evaluation episodes.

**Pro Tips to Follow:**
- Ensure inputs to the CNN are normalized ([0, 1]).
- Implement frame stacking correctly.
- Use fixed random seeds for reproducibility during development and debugging.
