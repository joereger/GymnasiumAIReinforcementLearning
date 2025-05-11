# Approaches: Pong (`ALE/PongNoFrameskip-v4`)

## Approach 1: Deep Q-Network (DQN) with "Smart Defaults"

**Algorithm:** Deep Q-Network (DQN)

**Rationale:** DQN is a standard and effective algorithm for pixel-based Atari environments like Pong. The "Smart Defaults" provided by the user offer a strong, battle-tested starting point.

**Source File(s):**
- `pong/pong_dqn.py` - Main entry point for the DQN implementation
- `pong/pong_dqn_utils.py` - Preprocessing, frame stacking, replay buffer utilities
- `pong/pong_dqn_model.py` - Neural network model and agent implementation
- `pong/pong_dqn_vis.py` - Visualization and plotting components
- `pong/pong_dqn_train.py` - Training and evaluation logic

**Key Components & Configuration (Based on "Smart Defaults" and subsequent enhancements):**

1.  **Environment Choice:**
    *   **Name:** `PongNoFrameskip-v4` (Corrected from `ALE/PongNoFrameskip-v4` as per Gymnasium naming for this variant)
    *   **Instantiation:** `env = gym.make("PongNoFrameskip-v4", repeat_action_probability=0.0)`
    *   **Reasoning:** Canonical choice for Atari DQN research, offers precise frame-skipping control via the agent.

2.  **Preprocessing Strategy (in `pong_dqn_utils.py`):**
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
        *   A `FrameStack` class is used (implemented in `pong_dqn_utils.py`).

3.  **CNN Model Structure (in `pong_dqn_model.py`, DeepMind 2015 Atari Paper):**
    *   **Input:** (4, 84, 84) - Stacked preprocessed frames.
    *   **Architecture:**
        1.  Conv1: 32 filters, 8x8 kernel, stride 4, ReLU activation. Output: (32, 20, 20)
        2.  Conv2: 64 filters, 4x4 kernel, stride 2, ReLU activation. Output: (64, 9, 9)
        3.  Conv3: 64 filters, 3x3 kernel, stride 1, ReLU activation. Output: (64, 7, 7)
        4.  Flatten
        5.  Dense: 512 units, ReLU activation.
        6.  Output: Linear layer with `action_space` units (typically 6 for Pong: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE; or simpler if environment action space is reduced).
    *   The `PongDQN` PyTorch module is implemented in `pong_dqn_model.py`.

4.  **Hyperparameters (Recommended Starting Point):**
    *   **Optimizer:** Adam
    *   **Learning Rate (Experiment 1):** `2.5e-4` (Initial default, also used for Experiment 1 after trying `1e-4`)
    *   **Batch Size:** `32`
    *   **Replay Buffer Size:** `1e5`
    *   **Replay Buffer Warmup:** `50,000` steps with random actions before training begins.
    *   **Discount Factor (γ):** `0.99`
    *   **Target Network Update Frequency (Experiment 1):** Every `10,000` agent learning steps (increased from 1000).
    *   **Exploration (ε-greedy):**
        *   Initial ε: `1.0`
        *   Final ε: `0.01`
        *   Decay Period: Over `1,000,000` agent steps.

5.  **DQN Agent Components & Enhancements (in `pong_dqn_model.py`):**
    *   **Q-Network (Online Network):** The `PongDQN` model described above.
    *   **Target Network:** A separate instance of `PongDQN`, with weights periodically copied from the online network (frequency adjusted in Experiment 1).
    *   **Replay Buffer:** A deque-based buffer of size `1e5`. Stores `(state, action, clipped_reward, next_state, done)` tuples.
    *   **Reward Clipping:** Rewards from the environment are clipped to `np.sign(reward)` (i.e., -1, 0, or 1) before being stored in the replay buffer and used for learning.
    *   **Gradient Clipping:** Gradients are clipped to a maximum norm of `1.0` during the optimizer step (`torch.nn.utils.clip_grad_norm_`).
    *   **Loss Function:** Mean Squared Error (MSE) between predicted Q-values and target Q-values.

6.  **Training Loop Structure & Logging (in `pong_dqn_train.py`):**
    *   Iterate for `max_episodes`.
    *   Inside each episode:
        *   Reset environment and frame stack.
        *   Loop until `done`:
            *   Agent selects action using ε-greedy policy based on current `state`.
            *   Take step in environment: `next_obs, reward, terminated, truncated, info`.
            *   Preprocess `next_obs` and update frame stack to get `next_state`.
            *   Store `(state, action, clipped_reward, next_state, done)` in replay buffer.
            *   Update `state = next_state`.
            *   Accumulate raw episode reward (for logging actual game score).
            *   Call `agent.learn()` (which includes gradient clipping) after each step (if buffer is full enough).
                *   Samples minibatch from replay buffer.
                *   Calculates target Q-values using the target network.
                *   Performs gradient descent step on the online network.
                *   Periodically updates the target network (frequency adjusted).
    *   **Evaluation:** Periodically (every 10 episodes) evaluate agent performance with ε=0. The best performing model based on these evaluations is saved.
    *   **Checkpointing:** Regular model checkpoints and detailed training statistics (including episode rewards, steps, epsilon, timestamps, durations, average loss, average max Q-value) are saved to a JSON file every 10 episodes. This allows for resumable training and plotting.
    *   **Plotting (in `pong_dqn_vis.py`):** Training progress (rewards, avg max Q, avg loss) is plotted from the saved statistics.

7.  **Code Organization Improvements:**
    *   Modular structure with clear separation of concerns
    *   Better maintainability through smaller, focused files
    *   Enhanced documentation and consistent style throughout
    *   Clean entry point with thorough dependency checking

**Current Status & Next Steps (Experiment 1):**
- Previous runs (up to ~500 episodes / 1.7M steps with LR `2.5e-4` then `1e-4`, and target update at 1000 steps) showed stagnation with average rewards around -20.5 and evaluation scores at -21.
- The code has been refactored into a modular structure while preserving all functionality.
- **Experiment 1 aims to address stagnation by:**
    - Reverting Learning Rate to `2.5e-4`.
    - Significantly increasing Target Network Update Frequency to `10,000` steps.
    - Continuing with reward clipping, 50k step replay buffer warmup, and gradient clipping.
    - Enhanced logging (avg loss, avg max Q) for better insight.
- The goal is to see if more stable targets and the original LR can help the agent break out of the performance plateau.

**Expected Outcome (Revised):**
With the applied fixes, code refactoring, and hyperparameter adjustments in Experiment 1, the hope is to see a sustained upward trend in the 100-episode average reward and, critically, an improvement in the greedy evaluation scores beyond -21.

**Pro Tips to Follow (Maintained):**
- Ensure inputs to the CNN are normalized ([0, 1]).
- Implement frame stacking correctly.
- Use fixed random seeds for reproducibility during development and debugging (for a given experimental run).
- Monitor new logging metrics (avg loss, avg max Q) for signs of learning stability or issues.
