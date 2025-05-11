# Progress: Pong (`ALE/PongNoFrameskip-v4`)

**Overall Status:** DQN agent implemented. Initial training runs showed stagnation. Currently preparing for "Experiment 1" with adjusted hyperparameters and enhanced logging.

**Current Phase:** Iterative Tuning and Observation.

**Completed Steps:**
1.  **Initial Setup:**
    *   Directory structure (`pong/`, `data/pong/`, `memory-bank/environments/pong/`) created.
    *   Initial Pong-specific Memory Bank files populated.
2.  **`pong/pong_dqn.py` Implementation:**
    *   Core DQN components implemented: `preprocess`, `FrameStack`, `PongDQN` model, `ReplayBuffer`, `DQNAgent`.
    *   Training loop, evaluation, plotting, and command-line interaction for train/evaluate choices.
    *   Robust JSON-based statistics saving/loading for resumable training and charts.
    *   Seed correction for warmup.
    *   Environment ID corrected to `PongNoFrameskip-v4`.
3.  **"High-Impact Fixes" Implemented:**
    *   Reward Clipping (`np.sign(reward)`).
    *   Replay Buffer Warmup (50,000 steps).
    *   Gradient Clipping (norm 1.0).
    *   Ensured `agent.current_frames` is correctly restored when resuming training.
4.  **Enhanced Logging & Plotting:**
    *   Added collection and logging of average max Q-value and average loss per episode.
    *   Plotting function updated to display these new metrics.
5.  **Initial Training Runs & Observation:**
    *   Run 1 (LR `2.5e-4`, Target Update `1k` steps): ~1.2M steps, ~9 hours. Result: Stagnation, avg reward ~-20.5, eval score -21.
    *   Run 2 (LR `1e-4`, Target Update `1k` steps): ~500 episodes, ~1.7M total steps from combined runs. Result: Continued stagnation, avg reward trended slightly worse to ~-20.6, eval score -21.

**What's Working:**
-   The `pong_dqn.py` script runs, trains, saves/loads checkpoints and stats, and generates multi-panel plots.
-   The core DQN architecture and training enhancements (clipping, warmup) are in place.

**What's Next (Experiment 1):**
1.  **Execute Fresh Training Run for Experiment 1:**
    *   **Learning Rate:** `2.5e-4`.
    *   **Target Network Update Frequency:** `10,000` steps.
    *   All other enhancements (reward clipping, 50k warmup, gradient clipping, robust stats, new logging) active.
    *   Ensure previous stats/models are cleared or not loaded to start fresh.
2.  **Monitor Closely:** Observe 100-episode average reward, evaluation scores, average max Q-values, and average loss.
3.  **Analyze Results:** Determine if the increased target update frequency leads to improved learning stability and performance.
4.  **Documentation:** Update this `progress.md` and other Memory Bank files with the outcomes of Experiment 1.
5.  **Further Iteration:** Based on Experiment 1 results, decide on subsequent hyperparameter adjustments or other diagnostic steps if stagnation persists.

**Known Issues/Challenges:**
-   **Persistent Learning Stagnation:** The primary challenge is that the agent has not shown significant learning despite extensive training and initial hyperparameter tweaks. Evaluation scores remain at the minimum (-21).
-   Identifying the root cause of non-learning (hyperparameters, subtle bug, exploration issues) is the main focus.

**Timeline Estimation (Rough for Experiment 1):**
-   Another significant training run (e.g., 1M-2M frames / ~10-20 hours) will be needed to assess the impact of the new target update frequency.
