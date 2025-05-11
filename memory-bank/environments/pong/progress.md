# Progress: Pong (`PongNoFrameskip-v4`)

**Overall Status:** DQN agent (Experiment 1) showed Q-value collapse. Double DQN (Experiment 2) implemented to address this issue.

**Current Phase:** Experiment 2 - Double DQN Implementation

**Completed Steps:**
1.  **Initial Setup:**
    *   Directory structure (`pong/`, `data/pong/`, `memory-bank/environments/pong/`) created.
    *   Initial Pong-specific Memory Bank files populated.

2.  **Original `pong/pong_dqn.py` Implementation (Experiment 1):**
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

5.  **Code Refactoring:**
    *   Refactored the monolithic `pong/pong_dqn.py` into a more maintainable modular structure:
        *   `pong_dqn.py` - Main entry point with dependency checking and user interface
        *   `pong_dqn_utils.py` - Preprocessing, frame stacking, replay buffer
        *   `pong_dqn_model.py` - DQN network architecture and agent implementation
        *   `pong_dqn_vis.py` - Visualization and multi-axis plotting
        *   `pong_dqn_train.py` - Training and evaluation logic
    *   All functionality preserved while improving code maintainability and readability
    *   Updated Memory Bank documentation to reflect the new structure

6.  **Experiment 1 Results Analysis:**
    *   Ran the DQN implementation for ~50 episodes (180K steps)
    *   Discovered concerning Q-value collapse: values steadily declined from ~0.053 to negative values
    *   Rewards stagnated between -21 and -19, with no improvement trend
    *   Loss remained stable around 0.006 throughout training
    *   No improvement in evaluation performance (consistently -21 score)
    *   Identified potential overestimation bias as a major contributor to learning issues

7.  **Double DQN Implementation (Experiment 2):**
    *   Created new set of files for Double DQN implementation:
        *   `pong_double_dqn.py` - Main Double DQN entry point
        *   `pong_double_dqn_utils.py` - Utilities (buffer size increased to 500K)
        *   `pong_double_dqn_model.py` - Double DQN algorithm implementation
        *   `pong_double_dqn_vis.py` - Enhanced visualization with experiment details
        *   `pong_double_dqn_train.py` - Modified training loop for Double DQN
    *   Implemented key algorithmic change: decoupling action selection (using online network) from action evaluation (using target network)
    *   Updated hyperparameters:
        *   Learning rate: Reduced to `1e-5` (from `2.5e-4`)
        *   Target network updates: More frequent at every `1,000` steps (from `10,000`)
        *   Replay buffer size: Increased to `500K` (from `100K`)
        *   Epsilon decay: Slower over `2M` frames (from `1M`)
    *   Improved training stats to include experiment version and hyperparameters
    *   Added experiment details to visualization

**What's Working:**
-   The codebase is now more maintainable with a modular structure
-   Comprehensive logging and visualization provides clear insights into training dynamics
-   Double DQN implementation follows research best practices for addressing Q-value overestimation
-   All training/evaluation functionality for both DQN and Double DQN is working correctly

**What's Next (Experiment 2):**
1.  **Execute Training Run with Double DQN:**
    *   Run the `pong_double_dqn.py` implementation for a significant number of frames (1-2M)
    *   Monitor Q-values to check if the collapse issue is addressed
    *   Track reward trends, especially evaluation scores
    *   Periodically check the plots to assess learning progress

2.  **Results Analysis:**
    *   Compare Double DQN to the original DQN implementation
    *   Assess impact of each hyperparameter change
    *   Document findings in Memory Bank

3.  **Next Steps After Experiment 2:**
    *   If Double DQN shows promise: pursue further enhancements (Prioritized Experience Replay, Dueling DQN)
    *   If Double DQN still struggles: investigate more fundamental issues (state representation, reward structure)
    *   Update the Memory Bank with new findings and next experiment plans

**Known Issues/Challenges:**
-   **Q-value Collapse in Original DQN:** The most critical issue in Experiment 1, with Q-values steadily declining to negative values
-   **Learning Stagnation:** Despite extensive training, the original DQN showed no improvement in rewards
-   **Fundamental Learning Dynamics:** Even with reward clipping, proper warmup, and gradient clipping, the learning dynamics appear unstable

**Timeline Estimation:**
-   Double DQN training run will likely require 1-2M frames (~10-20 hours) to show meaningful trends
-   Given the slower epsilon decay (2M frames), learning progress may be more gradual but potentially more stable
