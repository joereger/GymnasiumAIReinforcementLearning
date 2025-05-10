# Active Context: Pong (`ALE/PongNoFrameskip-v4`)

**Current Task:** Initial setup and implementation of a DQN agent for the Pong environment.

**Phase:** Environment Setup and Initial Memory Bank Population.

**Recent Actions:**
1.  Created directory structure:
    *   `pong/` (for solution code)
    *   `memory-bank/environments/pong/` (for Pong-specific Memory Bank files)
    *   `data/pong/` (for Pong-specific data)
2.  Created `memory-bank/environments/pong/environment_brief.md` detailing the `ALE/PongNoFrameskip-v4` environment.
3.  Created `memory-bank/environments/pong/approaches.md` outlining the DQN approach with "Smart Defaults" (environment, preprocessing, CNN model, hyperparameters).
4.  Created `memory-bank/environments/pong/systemPatterns.md` with Python code snippets for `preprocess`, `FrameStack`, `PongDQN` model, and `ReplayBuffer` based on "Smart Defaults".
5.  Created `memory-bank/environments/pong/techContext.md` detailing relevant technologies and libraries.

**Next Steps:**
1.  Create `memory-bank/environments/pong/progress.md` with initial status.
2.  Create the main Python script `pong/pong_dqn.py`. This script will include:
    *   Imports: `gymnasium`, `torch`, `numpy`, `cv2`, `random`, `collections.deque`, `time`, `matplotlib.pyplot`.
    *   Device configuration (CPU/GPU).
    *   `preprocess` function.
    *   `FrameStack` class.
    *   `PongDQN` neural network class.
    *   `ReplayBuffer` class.
    *   `DQNAgent` class (or a set of agent functions) incorporating:
        *   Initialization (Q-network, target network, optimizer, replay buffer, epsilon parameters).
        *   `act` method (for ε-greedy action selection).
        *   `learn` method (for sampling from buffer and training the Q-network).
        *   Target network update logic.
    *   Hyperparameter definitions.
    *   Training loop.
    *   Evaluation function.
    *   Plotting function for rewards.
    *   Main execution block (`if __name__ == "__main__":`).
3.  Create/copy `pong/visualization_utils.py` (if distinct plotting functions are preferred, otherwise integrate into `pong_dqn.py`). For now, plotting will be integrated into `pong_dqn.py`.
4.  Update top-level Memory Bank files (`activeContext.md`, `progress.md`) to reflect the new Pong environment work.

**Open Questions/Decisions:**
-   Action space for Pong: The `PongDQN` model in `systemPatterns.md` defaults to `action_space=6`. Need to confirm the actual action space of `ALE/PongNoFrameskip-v4` from Gymnasium and adjust if necessary. (Commonly, Pong uses a reduced action set like 2 or 3 actions: UP, DOWN, [STAY/FIRE]). This will be checked when `pong_dqn.py` is created.

**Focus:** Adhering to the "Smart Defaults" and project conventions for a robust initial implementation.
