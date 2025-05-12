# Pong Active Context

## Current Status (Last Updated: 5/11/2025)

We are focusing on a **barebones DQN implementation** for Pong to establish a working baseline. Key changes include:

1.  **Reduced Action Space**: The environment now uses only 3 actions (STAY, PADDLE_UP, PADDLE_DOWN), which is critical for efficient learning in Pong. This is handled by the `ReducedActionSpace` wrapper in `pong_dqn_utils.py`.
2.  **Simplified Hyperparameters**:
    *   `REPLAY_BUFFER_SIZE` reduced to 50,000.
    *   `TARGET_UPDATE_FREQ` reduced to 1,000 frames.
    *   `EPSILON_DECAY_FRAMES` reduced to 100,000 frames.
    *   `MAX_FRAMES_TOTAL` reduced to 500,000 for faster test cycles.
3.  **No Replay Buffer Warmup**: The lengthy warmup phase has been removed. Learning (calling `agent.learn()`) will begin as soon as the replay buffer has `BATCH_SIZE` experiences.
4.  **Consistent Tensor Handling**: Ensured that the `StackFrame` wrapper correctly provides `(4, 84, 84)` channels-first, normalized frames to the DQN.
5.  **Evaluation Epsilon**: The `evaluate_pong_agent` function now uses a small `eval_epsilon` (default 0.05) to allow for some exploration during evaluation, giving a more realistic performance measure than a purely greedy policy, especially for partially trained agents.

## Next Steps

*   Run the simplified `pong_dqn_train.py` script.
*   Monitor initial learning progress. We expect to see *some* improvement in scores over a reasonable number of frames (e.g., within the first 100k-200k frames), even if it doesn't reach expert level quickly.
*   If learning still doesn't occur, the next step will be to meticulously debug the `DQNAgent.learn()` method and the loss calculation.

## Outstanding Concerns

*   The previous issue where the first episode showed a decent score and then performance plummeted needs to be monitored. If it recurs even with a fresh start (no loaded checkpoints) and the simplified setup, it points to a very subtle bug in the learning update or state representation.
