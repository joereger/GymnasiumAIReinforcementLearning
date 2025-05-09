# Approaches: Cart Pole

This document details the different solution approaches implemented or explored for the Cart Pole environment.

## Approach 1: Deep Q-Network (DQN) with Enhanced Reward Engineering

- **Source File:** `cart_pole/cart_pole.py`
- **Algorithm:** Deep Q-Network (DQN) with replay memory
- **Key Hyperparameters & Configuration:**
    - **Neural Network Architecture:**
        - Input layer: 4 nodes (state dimensions)
        - Hidden layers: 512 nodes → 256 nodes → 64 nodes (all ReLU activation)
        - Output layer: 2 nodes (linear activation for Q-values of actions)
    - **Learning Parameters:**
        - Optimizer: RMSprop (learning_rate=0.00025, rho=0.95, epsilon=0.01)
        - Discount factor (gamma): 0.95
        - Initial exploration rate (epsilon): 1.0
        - Epsilon decay: 0.999
        - Epsilon minimum: 0.001
        - Batch size: 64
        - Replay memory size: 2000 experiences
        - Training starts after: 1000 experiences collected
    - **Reward Engineering Components:**
        - **Center Comfort Zone Bonus:** +0.25 at center, tapering to 0 at distance ±0.5
        - **Progressive Position Penalties:**
            - Close to center (< 0.8): -0.5 * (position^2)
            - Middle region (< 1.5): -1.0 * (position^2)
            - Outer regions (≥ 1.5): -2.0 * (position^2)
        - **Directional Velocity Penalties:**
            - Moving away from center: -3.0 * (velocity^2), increased to -4.0 when position > 1.0
            - Moving toward center: -0.3 * (velocity^2)
            - Danger zone multiplier: 1.5x penalty for high velocity away from center when position > 1.2
        - **Recovery Bonuses:**
            - Base recovery: Position-scaled reward for any movement toward center
            - Medium recovery (position > 1.0): 3.0x bonus multiplier
            - Major recovery (position > 1.5): 5.0x bonus multiplier
        - **Edge Avoidance:** -10.0 * (position - 2.0) penalty when position > 2.0
    - **Episode Termination:** Force episode termination and save model when score reaches 500 steps

- **Results & Performance:**
    - **Success Rate:** Consistently solved the environment (500+ steps)
    - **Position Control:** Successfully learned to keep cart near center while balancing pole
    - **Stability:** Achieved perfect stability around position -0.2, balancing indefinitely
    - **Recovery Skills:** Demonstrated ability to recover from edge positions when destabilized
    - **Data Persistence:** Models saved to environment-specific directory (`data/cart_pole/`)

- **Learnings & Insights:**
    - **Reward Engineering Impact:** The multi-component reward system proved extremely effective, teaching the agent to prioritize both pole balance and position control simultaneously.
    - **Critical Components:**
        - The directional velocity penalties were particularly important, preventing the cart from building up dangerous momentum when moving away from center.
        - Progressive penalties created the necessary gradient for learning centrality.
        - The recovery bonus system encouraged corrective behavior when drifting too far.
    - **Balancing vs. Centrality Tradeoff:** The default CartPole environment only rewards keeping the pole upright, leading to solutions that drift to the edges. Our reward engineering successfully balanced the dual objectives.
    - **Challenge Resolution:** The agent became so effective that it learned to stabilize indefinitely, requiring explicit episode termination to recognize success.
    - **Data Organization:** Confirmed the importance of environment-specific data directories for model persistence.

## Potential Future Approaches

1. **Proximal Policy Optimization (PPO):** Could explore using this policy-based method which is known for stable learning on continuous control tasks.

2. **Actor-Critic Methods:** A hybrid approach that could potentially learn more efficiently by reducing variance.

3. **Simplified Reward Engineering:** Test whether simpler reward formulations could achieve similar results with less complexity.

4. **Domain Randomization:** Introducing randomness in initial conditions and physics parameters to build more robust controllers.
