# Active Context: Pong (`PongNoFrameskip-v4`)

**Current Task:** Implementing Double DQN to address Q-value collapse in the initial DQN implementation.

**Phase:** Experiment 2 - Double DQN Implementation

**Recent Actions & State:**
1. **Code Modularization:**
   * Refactored the original monolithic `pong_dqn.py` into a modular structure:
     * `pong_dqn.py` - Main entry point
     * `pong_dqn_utils.py` - Preprocessing and utilities
     * `pong_dqn_model.py` - DQN network and agent
     * `pong_dqn_vis.py` - Visualization
     * `pong_dqn_train.py` - Training and evaluation

2. **Experiment 1 Results Analysis:**
   * Ran the DQN implementation for ~50 episodes (180K steps)
   * Observed persistent stagnation with rewards between -21 and -19
   * Most concerning: **Q-value collapse** from ~0.053 to negative values
   * Loss remained stable (~0.006) despite declining Q-values
   * No improvement in evaluation performance (consistently -21 score)
   * These issues suggest potential overestimation bias and learning instability

3. **Double DQN Implementation (Experiment 2):**
   * Created a new set of files with Double DQN implementation:
     * `pong_double_dqn.py`
     * `pong_double_dqn_utils.py`
     * `pong_double_dqn_model.py`
     * `pong_double_dqn_vis.py`
     * `pong_double_dqn_train.py`
   * **Key Algorithm Change:** Decoupled action selection and evaluation
     * Using online network to select actions: `best_action = argmax(Q_online(next_state))`
     * Using target network to evaluate actions: `target_q = reward + gamma * Q_target(next_state, best_action)`
   * **Hyperparameter Changes:**
     * Learning rate: Reduced to `1e-5` (from `2.5e-4`)
     * Target network updates: More frequent at every `1,000` steps (from `10,000`)
     * Replay buffer size: Increased to `500K` (from `100K`)
     * Epsilon decay: Slower over `2M` frames (from `1M`)
   * **Enhanced Visualization:** Added experiment information to plots, improved metrics display

**Next Steps:**
1. **Run Experiment 2:**
   * Execute a fresh training run with the Double DQN implementation
   * Monitor Q-values closely to check if the collapse issue is addressed
   * Pay attention to reward trends, especially evaluation scores
   * Allow sufficient training time (1-2M frames) to assess long-term learning trends

2. **Analysis:**
   * Compare Double DQN performance to original DQN implementation
   * Look for stability in Q-values (no collapse) and gradual improvement in rewards
   * Document the impact of each hyperparameter change

3. **Future Considerations:**
   * If Double DQN shows promise but still struggles, consider:
     * Prioritized Experience Replay
     * Dueling DQN architecture
     * Further hyperparameter adjustments
   * If Double DQN fails completely, investigate more fundamental issues:
     * State representation (possibly RAM-based state)
     * Reward structure (potential reward shaping)
     * Network architecture changes

**Open Questions/Decisions:**
- Will Double DQN address the Q-value collapse observed in the original implementation?
- Which hyperparameter change will have the biggest impact: learning rate, target update frequency, or replay buffer size?
- Is the main issue overestimation bias (which Double DQN addresses) or something more fundamental?

**Focus:** 
Determining if Double DQN with adjusted hyperparameters can overcome the learning stagnation observed in the vanilla DQN implementation.
