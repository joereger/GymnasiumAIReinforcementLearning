# Active Context: Pong (`ALE/PongNoFrameskip-v4`)

**Current Task:** Experimenting with DQN hyperparameters and enhancements to achieve learning in the Pong environment.

**Phase:** Iterative Tuning and Observation (Experiment 1).

**Recent Actions & State:**
1.  **Initial DQN Implementation:**
    *   Created `pong/pong_dqn.py` with standard DQN components (CNN, Replay Buffer, Target Network, Frame Stacking, Epsilon-Greedy).
    *   Initial hyperparameters based on "Smart Defaults" (LR: `2.5e-4`, Target Update: `1000` steps).
2.  **Observed Stagnation:** Training runs (up to ~500 episodes / 1.7M steps, including a test with LR `1e-4`) showed no significant learning. Average rewards remained around -20.5, and evaluation scores were consistently -21.
3.  **Implemented "High-Impact Fixes":**
    *   **Reward Clipping:** Rewards stored in the replay buffer are clipped to `np.sign(reward)`.
    *   **Replay Buffer Warmup:** Added a 50,000-step warmup phase.
    *   **Gradient Clipping:** Enabled gradient norm clipping at 1.0 in `agent.learn()`.
    *   **Corrected `agent.current_frames` Resumption:** Ensured `agent.current_frames` is properly restored from stats when loading a checkpoint.
    *   **Corrected Environment ID:** Using `PongNoFrameskip-v4`.
    *   **Corrected Warmup Seeding:** Ensured positive seeds during warmup resets.
4.  **Enhanced Logging & Plotting:**
    *   Added logging for average max Q-value and average loss per episode to console and JSON stats.
    *   Updated plotting function to display these new metrics alongside rewards.
5.  **Initiated Experiment 1:**
    *   **Learning Rate:** `2.5e-4` (reverted to original default).
    *   **Target Network Update Frequency:** Increased to `10,000` steps.
    *   All other enhancements (reward clipping, 50k warmup, gradient clipping, robust stats) are active.

**Next Steps (Focus of Experiment 1):**
1.  Execute a **fresh training run** of `pong/pong_dqn.py` with the Experiment 1 settings (LR `2.5e-4`, Target Update `10k` steps).
    *   Ensure previous `data/pong/pong_training_stats.json` and model files are cleared or that "load checkpoint" is answered with 'n' to avoid contamination from previous runs.
2.  Monitor training progress closely, paying attention to:
    *   100-episode average reward.
    *   **Evaluation scores (every 10 episodes).** This is the most critical indicator.
    *   Average max Q-values per episode (should ideally trend upwards if value estimation is improving).
    *   Average loss per episode (should ideally decrease and stabilize).
3.  Run for a significant number of frames (e.g., 1M to 2M) to give the new settings a fair chance.
4.  Analyze the results of Experiment 1 to determine if these changes have led to any improvement in learning.
5.  Update Pong-specific Memory Bank files (`approaches.md`, `systemPatterns.md`, `activeContext.md`, `progress.md`) with the details and outcomes of Experiment 1.
6.  Update top-level Memory Bank files if significant project-level insights are gained.

**Open Questions/Decisions:**
-   Will the increased target network update frequency, combined with the original learning rate and other enhancements, be sufficient to break the learning stagnation?
-   If Experiment 1 fails, what will be the next set of hyperparameters or structural changes to investigate (e.g., even lower learning rate with slow target updates, different optimizer, or considering RAM-based state)?

**Focus:** Systematically testing the impact of a slower target network update frequency on learning stability and performance, while leveraging enhanced observability through new logging metrics.
