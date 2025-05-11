# Approaches: Pong (`PongNoFrameskip-v4`)

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

4.  **Hyperparameters (Experiment 1):**
    *   **Optimizer:** Adam
    *   **Learning Rate:** `2.5e-4`
    *   **Batch Size:** `32`
    *   **Replay Buffer Size:** `1e5`
    *   **Replay Buffer Warmup:** `50,000` steps with random actions before training begins.
    *   **Discount Factor (γ):** `0.99`
    *   **Target Network Update Frequency:** Every `10,000` agent learning steps (increased from the initial 1,000).
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

**Results (Experiment 1):**
- Training ran for ~50 episodes (180K steps)
- Rewards stagnated between -21 and -19, with no improvement trend
- 100-episode average reward remained around -20.4
- Q-values showed a steady decline from ~0.053 to negative values
- Loss remained stable around 0.006 throughout training
- Best evaluation reward achieved was -21.0 (worst possible score)
- Showed potential Q-value collapse or divergence, indicating learning issues

## Approach 2: Double DQN (Experiment 2)

**Algorithm:** Double Deep Q-Network (Double DQN)

**Rationale:** The original DQN implementation (Experiment 1) showed Q-value collapse, suggesting potential overestimation bias. Double DQN addresses this by decoupling action selection from action evaluation.

**Source File(s):**
- `pong/pong_double_dqn.py` - Main entry point for the Double DQN implementation
- `pong/pong_double_dqn_utils.py` - Preprocessing, frame stacking, replay buffer utilities
- `pong/pong_double_dqn_model.py` - Neural network model and Double DQN agent implementation
- `pong/pong_double_dqn_vis.py` - Visualization and plotting components
- `pong/pong_double_dqn_train.py` - Training and evaluation logic

**Key Components & Configuration:**

1. **Environment Choice:**
   * Same as Approach 1: `PongNoFrameskip-v4` with `repeat_action_probability=0.0`

2. **Preprocessing Strategy:**
   * Identical to Approach 1 (grayscale conversion, downsampling to 84x84, normalization to [0,1], frame stacking)

3. **CNN Model Structure:**
   * Identical architecture to Approach 1 (DeepMind 2015 Atari paper)

4. **Hyperparameters (Experiment 2):**
   * **Optimizer:** Adam
   * **Learning Rate:** `1e-5` (reduced from 2.5e-4 in Experiment 1)
   * **Batch Size:** `32` (unchanged)
   * **Replay Buffer Size:** `5e5` (increased from 1e5 in Experiment 1)
   * **Replay Buffer Warmup:** `50,000` steps (unchanged)
   * **Discount Factor (γ):** `0.99` (unchanged)
   * **Target Network Update Frequency:** Every `1,000` agent learning steps (more frequent than 10,000 in Experiment 1)
   * **Exploration (ε-greedy):**
      * Initial ε: `1.0` (unchanged)
      * Final ε: `0.01` (unchanged)
      * Decay Period: Over `2,000,000` agent steps (slower than 1,000,000 in Experiment 1)

5. **Double DQN Algorithm:**
   * **Core Difference:** Double DQN modifies how target Q-values are calculated:
     * **Standard DQN:** `target_q = reward + gamma * max(Q_target(next_state))`
     * **Double DQN:** 
       1. Uses online network to SELECT actions: `best_action = argmax(Q_online(next_state))`
       2. Uses target network to EVALUATE those actions: `target_q = reward + gamma * Q_target(next_state, best_action)`
   * This decoupling reduces overestimation bias by preventing the target network from both selecting and evaluating the actions

6. **Enhanced Visualization:**
   * Multi-axis plots showing rewards, average max Q-values, and average loss
   * Embedded experiment parameters in the plot for easy reference
   * Horizontal line at reward 0 to highlight when the agent starts winning

7. **Improved Stats Tracking:**
   * Tracks experiment version and hyperparameters in the stats file
   * More detailed logging during training

**Results (Experiment 2):**
- Training ran for 126 episodes (~438K steps) before process termination
- Rewards continued to stagnate between -21 and -18
- 100-episode moving average remained around -20.5 
- Q-values still showed a concerning decline from +0.008 to approximately -0.49 by episode 120
- Loss decreased more gradually (from ~0.006 to ~0.001) than in Experiment 1
- No improvement in evaluation performance
- Double DQN failed to address the Q-value collapse issue observed in vanilla DQN
- Despite algorithm changes and hyperparameter adjustments, no signs of successful learning

## Approach 3: Proximal Policy Optimization (PPO) (Experiment 3)

**Algorithm:** Proximal Policy Optimization (PPO)

**Rationale:** After failures with both DQN and Double DQN (value-based methods), shifting to a policy gradient approach like PPO may better handle the sparse rewards in Pong. PPO offers more stable learning with its clipped surrogate objective and has shown excellent performance on Atari games in literature.

**Source File(s)** (to be implemented):
- `pong/pong_ppo.py` - Main entry point for the PPO implementation
- `pong/pong_ppo_agent.py` - PPO agent implementation
- `pong/pong_ppo_model.py` - Actor-Critic network architecture
- `pong/pong_ppo_train.py` - Training loop and rollout collection
- `pong/pong_ppo_utils.py` - Preprocessing and utilities
- `pong/pong_ppo_vis.py` - Visualization tools

**Key Components & Configuration:**

1. **Environment Choice:**
   * Same as previous approaches: `PongNoFrameskip-v4` with `repeat_action_probability=0.0`

2. **Preprocessing Strategy:**
   * Identical to previous approaches (grayscale conversion, downsampling to 84x84, normalization to [0,1], frame stacking)

3. **Actor-Critic Architecture:**
   * **Feature Extractor:** Shared CNN layers similar to DQN architecture
     * Conv1: 32 filters, 8x8 kernel, stride 4, ReLU activation
     * Conv2: 64 filters, 4x4 kernel, stride 2, ReLU activation
     * Conv3: 64 filters, 3x3 kernel, stride 1, ReLU activation
     * Flatten
   * **Actor (Policy) Network:**
     * Dense: 512 units, ReLU activation
     * Output: Softmax layer over actions to output action probabilities
   * **Critic (Value) Network:**
     * Dense: 512 units, ReLU activation
     * Output: Single scalar value estimate

4. **PPO Specific Components:**
   * **Clipped Surrogate Objective:** Limit the policy update to prevent too large changes
   * **Generalized Advantage Estimation (GAE):** Calculate advantages with weighted returns for variance reduction
   * **Multiple Epochs:** Perform several optimization passes over each batch of collected experience
   * **Entropy Bonus:** Encourage exploration by adding policy entropy to the loss function
   * **On-Policy Learning:** Train only on data collected from the current policy

5. **Hyperparameters (Planned):**
   * **Optimizer:** Adam
   * **Learning Rate:** 3e-4 (typical for PPO in Atari)
   * **Clip Parameter (ε):** 0.2 (balance between exploration and exploitation)
   * **GAE Parameter (λ):** 0.95 (trade-off between bias and variance)
   * **Value Function Coefficient:** 0.5 (balances value loss importance)
   * **Entropy Coefficient:** 0.01 (encourages exploration)
   * **Discount Factor (γ):** 0.99 (same as previous approaches)
   * **Epochs Per Update:** 4 (multiple passes through the same data)
   * **Rollout Length:** 128 steps (length of trajectories before updating)
   * **Minibatch Size:** 32 (for SGD updates)

6. **Data Collection:**
   * Collect rollouts of fixed length (e.g., 128 steps)
   * Store states, actions, rewards, values, log probabilities, dones
   * Process rollouts to compute advantages and returns
   * Perform multiple epochs of optimization on minibatches

7. **Monitoring & Visualization:**
   * Track episode rewards and lengths
   * Monitor policy entropy to ensure adequate exploration
   * Track value loss and policy loss separately
   * Visualize learning curves with multi-axis plots

**Expected Advantages over DQN-based Approaches:**
1. **Better Exploration:** Stochastic policy naturally encourages exploration
2. **More Sample Efficient:** Policy gradient methods often learn from fewer samples
3. **No Q-value Collapse:** Not susceptible to Q-function overestimation/divergence 
4. **Better with Sparse Rewards:** Value function provides clearer learning signal
5. **Stability:** Clipped objective prevents destructive policy updates

**Implementation Plan:**
1. First implement the core PPO algorithm components
2. Create the Actor-Critic network architecture
3. Build rollout collection mechanism
4. Implement advantage calculation with GAE
5. Create the PPO update procedure
6. Add monitoring and visualization components
7. Run training with evaluation checkpoints
