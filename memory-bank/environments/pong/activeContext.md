# Active Context: Pong (`PongNoFrameskip-v4`)

**Current Task:** Implementing Proximal Policy Optimization (PPO) after two failed DQN-based approaches.

**Phase:** Experiment 3 - PPO Implementation

**Recent Actions & State:**
1. **Experiment 1 (Vanilla DQN) Results:**
   * Ran DQN implementation for ~50 episodes (180K steps)
   * Observed Q-value collapse from ~0.053 to negative values
   * Rewards stagnated between -21 and -19, with no improvement
   * No success in evaluation (consistently -21 score)

2. **Experiment 2 (Double DQN) Results:**
   * Implemented Double DQN to address potential overestimation bias
   * Ran for 126 episodes (~438K steps) before process termination
   * Applied hyperparameter changes:
     * Learning rate: Reduced to 1e-5 (from 2.5e-4)
     * Target network updates: More frequent at 1,000 steps (from 10,000)
     * Replay buffer size: Increased to 500K (from 100K)
     * Epsilon decay: Slower over 2M frames (from 1M)
   * Observed continued Q-value collapse (from +0.008 to -0.49)
   * Rewards still stagnated between -21 and -18
   * Loss decreased more gradually (from ~0.006 to ~0.001)
   * No improvement in performance despite algorithm enhancements

3. **Analysis of DQN-Based Approaches:**
   * Both vanilla DQN and Double DQN failed to learn effective policies
   * Q-value collapse persisted despite addressing overestimation bias
   * Issues may be more fundamental:
     * Sparse reward structure in Pong
     * Challenges in state representation
     * Possible architectural limitations

4. **Current Task: PPO Implementation (Experiment 3):**
   * Shifting from value-based to policy gradient approaches
   * Implementing Proximal Policy Optimization with:
     * Actor-Critic architecture
     * Clipped surrogate objective
     * Generalized Advantage Estimation (GAE)
     * On-policy learning with multiple optimization epochs
   * Creating a similar modular code structure for maintainability

**Next Steps:**
1. **Implement PPO Architecture:**
   * Setup Actor-Critic networks
   * Implement clipped surrogate objective function
   * Create trajectory collection mechanism
   * Setup advantage estimation with GAE

2. **Run Experiment 3:**
   * Execute PPO training for Pong
   * Monitor key metrics (episode rewards, policy entropy, value loss)
   * Compare performance against previous DQN approaches

3. **Analysis:**
   * Assess if policy gradient approach overcomes limitations of DQN
   * Determine if PPO handles sparse rewards more effectively
   * Evaluate stability of learning process

**Open Questions/Decisions:**
- Will PPO's stochastic policy exploration approach handle the exploration challenges better than ε-greedy?
- How will the Actor-Critic architecture compared to the single Q-network of DQN impact learning?
- Will policy gradient methods be more effective for the sparse reward structure of Pong?
- What PPO hyperparameters will be most impactful for Pong (clip parameter, GAE lambda, etc.)?

**Focus:** 
Implementing and evaluating if a policy gradient approach (PPO) can overcome the learning challenges encountered with DQN-based methods in the Pong environment.
