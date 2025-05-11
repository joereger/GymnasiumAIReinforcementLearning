# Active Context: Pong (`ALE/PongNoFrameskip-v4`)

**Current Task:** Experimenting with DQN hyperparameters and enhancements to achieve learning in the Pong environment.

**Phase:** Code Refactoring and Continuing Experiment 1

**Recent Actions & State:**
1. **Code Refactoring:**
   * Refactored the monolithic `pong/pong_dqn.py` into a more modular structure:
     * `pong_dqn.py` - Clean main entry point with dependency checking
     * `pong_dqn_utils.py` - Preprocessing, frame stacking, replay buffer
     * `pong_dqn_model.py` - DQN network architecture and agent implementation
     * `pong_dqn_vis.py` - Visualization with multi-axis plotting
     * `pong_dqn_train.py` - Training and evaluation logic
   * This modular approach improves maintainability and readability
   * All functionality has been preserved from the original implementation

2. **Previous Enhancements (Maintained):**
   * **Reward Clipping:** Rewards stored in the replay buffer are clipped to `np.sign(reward)`.
   * **Replay Buffer Warmup:** Added a 50,000-step warmup phase.
   * **Gradient Clipping:** Enabled gradient norm clipping at 1.0 in `agent.learn()`.
   * **Corrected `agent.current_frames` Resumption:** Ensured `agent.current_frames` is properly restored from stats when loading a checkpoint.
   * **Corrected Environment ID:** Using `PongNoFrameskip-v4`.
   * **Corrected Warmup Seeding:** Ensured positive seeds during warmup resets.

3. **Enhanced Logging & Plotting (Improved):**
   * Maintained logging for average max Q-value and average loss per episode
   * Enhanced plotting function with multi-axis display in `pong_dqn_vis.py`

4. **Experiment 1 Settings (Unchanged):**
   * **Learning Rate:** `2.5e-4` (original default)
   * **Target Network Update Frequency:** `10,000` steps
   * All other enhancements (reward clipping, 50k warmup, gradient clipping, robust stats) are active.

**Next Steps:**
1. Execute a **fresh training run** with the refactored code but maintaining Experiment 1 settings (LR `2.5e-4`, Target Update `10k` steps).
   * Ensure previous `data/pong/pong_training_stats.json` and model files are cleared or that "load checkpoint" is answered with 'n' to avoid contamination.

2. Monitor training progress closely, paying attention to:
   * 100-episode average reward.
   * **Evaluation scores (every 10 episodes).** This is the most critical indicator.
   * Average max Q-values per episode (should ideally trend upwards if value estimation is improving).
   * Average loss per episode (should ideally decrease and stabilize).

3. Run for a significant number of frames (1M to 2M) to give the settings a fair chance.

4. Analyze the results of Experiment 1 to determine if these changes have led to any improvement in learning.

5. If needed, implement Experiment 2 with adjusted hyperparameters based on Experiment 1 results.

**Open Questions/Decisions:**
- Will the refactored code structure maintain the same performance characteristics as the original?
- Will the increased target network update frequency, combined with the original learning rate and other enhancements, be sufficient to break the learning stagnation?
- If Experiment 1 fails, what will be the next set of hyperparameters or structural changes to investigate (e.g., even lower learning rate with slow target updates, different optimizer, or considering RAM-based state)?

**Focus:** 
Running Experiment 1 with the newly refactored code to verify both the code integrity and the effectiveness of the hyperparameter settings in learning the Pong environment.
