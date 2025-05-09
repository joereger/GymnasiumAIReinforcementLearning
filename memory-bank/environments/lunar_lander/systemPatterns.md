# System Patterns: Lunar Lander Solutions

This document outlines system-level patterns, architectural choices, and common structures observed or employed specifically within the solutions developed for the Lunar Lander environment.

**Common Patterns in Lunar Lander Solutions (`lunar_lander.py`):**
- **Agent Class Structure:** (Describe the agent's structure, e.g., methods for `select_action`, `train_step`, `update_target_network` if applicable).
- **Neural Network Architectures:**
    - For DQN (discrete): Typical architecture for the Q-network.
    - For Actor-Critic (continuous): Architectures for actor and critic networks.
    - Common choices for layers, activation functions.
- **Replay Buffer:** (Implementation details if a replay buffer is used, e.g., PER - Prioritized Experience Replay).
- **Exploration Strategy:**
    - For DQN: Epsilon-greedy or noisy networks.
    - For Actor-Critic: Exploration noise added to actions (e.g., Ornstein-Uhlenbeck noise or Gaussian noise).
- **Target Networks:** (Use and update strategy for target networks if employed, e.g., soft updates, periodic hard updates).
- **Learning Loop:** (Structure of interaction, data storage, sampling from buffer, and network updates).

*(This file will be updated as common patterns are identified or refined from the `lunar_lander.py` solution.)*
