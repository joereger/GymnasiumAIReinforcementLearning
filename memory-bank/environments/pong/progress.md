# Progress: Pong (`PongNoFrameskip-v4`)

**Overall Status:** Experiment 1 (DQN) and Experiment 2 (Double DQN) both failed to learn effective policies. Now implementing Experiment 3 with Proximal Policy Optimization (PPO).

**Current Phase:** Experiment 3 - PPO Implementation

**Completed Steps:**
1.  **Initial Setup:**
    *   Directory structure (`pong/`, `data/pong/`, `memory-bank/environments/pong/`) created.
    *   Initial Pong-specific Memory Bank files populated.

2.  **Experiment 1 (Vanilla DQN):**
    *   Core DQN components implemented in modular structure:
        *   `pong_dqn.py` - Main entry point with dependency checking and user interface
        *   `pong_dqn_utils.py` - Preprocessing, frame stacking, replay buffer
        *   `pong_dqn_model.py` - DQN network architecture and agent implementation
        *   `pong_dqn_vis.py` - Visualization and multi-axis plotting
        *   `pong_dqn_train.py` - Training and evaluation logic
    *   Implemented common enhancements:
        *   Reward Clipping (`np.sign(reward)`)
        *   Replay Buffer Warmup (50,000 steps)
        *   Gradient Clipping (norm 1.0)
    *   Hyperparameters:
        *   Learning Rate: `2.5e-4`
        *   Target Network Update: Every `10,000` steps
        *   Replay Buffer Size: `100K`
        *   Epsilon Decay: Over `1M` frames
    *   **Results:**
        *   Ran for ~50 episodes (180K steps)
        *   Q-values collapsed from +0.053 to negative values
        *   Rewards stagnated between -21 and -19
        *   No improvement in evaluation performance (consistently -21 score)

3.  **Experiment 2 (Double DQN):**
    *   Created new set of files for Double DQN implementation:
        *   `pong_double_dqn.py` - Main Double DQN entry point
        *   `pong_double_dqn_utils.py` - Utilities (buffer size increased to 500K)
        *   `pong_double_dqn_model.py` - Double DQN algorithm implementation
        *   `pong_double_dqn_vis.py` - Enhanced visualization with experiment details
        *   `pong_double_dqn_train.py` - Modified training loop for Double DQN
    *   Implemented algorithmic change: decoupling action selection (using online network) from action evaluation (using target network)
    *   Updated hyperparameters:
        *   Learning rate: Reduced to `1e-5` (from `2.5e-4`)
        *   Target network updates: More frequent at every `1,000` steps (from `10,000`)
        *   Replay buffer size: Increased to `500K` (from `100K`)
        *   Epsilon decay: Slower over `2M` frames (from `1M`)
    *   **Results:**
        *   Ran for 126 episodes (~438K steps) before process termination
        *   Q-values still collapsed from +0.008 to -0.49 by episode 120
        *   Rewards continued to stagnate between -21 and -18
        *   100-episode moving average remained around -20.5
        *   Loss decreased more gradually (from ~0.006 to ~0.001)
        *   No improvement in performance despite addressing overestimation bias

4.  **Analysis of DQN-Based Approaches:**
    *   Both vanilla DQN and Double DQN suffered from Q-value collapse
    *   Addressing overestimation bias did not solve the underlying issues
    *   Adjusting hyperparameters failed to yield improvements
    *   Problems may be more fundamental:
        *   Sparse reward structure of Pong
        *   Limitations in state representation
        *   Exploration difficulties in large state spaces

**What's Working:**
-   Modular code structure and robust training/evaluation pipeline
-   Data collection, visualization, and experiment tracking
-   Methodology for systematically testing different approaches

**What's Not Working:**
-   DQN-based approaches (both vanilla and Double DQN) failing to learn
-   Q-value collapse persisting despite algorithmic improvements
-   Epsilon-greedy exploration strategy not finding effective trajectories

**What's Next (Experiment 3 - PPO):**
1.  **Implement PPO Architecture:**
    *   Create a new set of files for the PPO implementation
    *   Develop Actor-Critic network architecture
    *   Implement PPO-specific components:
        *   Clipped surrogate objective
        *   Generalized Advantage Estimation (GAE)
        *   Trajectory collection and minibatch optimization
        *   Policy entropy bonus for exploration

2.  **Hyperparameter Selection:**
    *   Clip parameter (ε): 0.1-0.3
    *   GAE parameter (λ): 0.9-0.99
    *   Value function coefficient: 0.5-1.0
    *   Entropy coefficient: ~0.01
    *   Learning rate: 2-5e-4 (with potential annealing)
    *   Epochs per update: 3-10
    *   Rollout length: 128-2048 steps

3.  **Training and Evaluation:**
    *   Execute PPO training to see if it overcomes limitations of DQN
    *   Monitor key metrics (rewards, policy entropy, value loss)
    *   Conduct periodic evaluation to assess policy improvement

**Known Issues/Challenges:**
-   **Q-value Collapse:** Both DQN and Double DQN suffered from this issue
-   **Learning Stagnation:** Neither approach showed signs of policy improvement
-   **Sparse Rewards:** Pong only provides rewards when points are scored, creating a difficult credit assignment problem
-   **Complex Dynamics:** The precise timing and positioning required for Pong may be difficult to learn from pixel observations alone

**Timeline Estimation:**
-   PPO implementation: 1-2 days
-   Initial training run: 1-2 days for 1-2M frames
-   Further hyperparameter tuning if needed: 3-5 days
