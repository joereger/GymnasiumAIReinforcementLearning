# Tech Context: Pong (`ALE/PongNoFrameskip-v4`)

This document outlines the key technologies, libraries, and technical considerations specific to implementing the DQN solution for the Pong environment.

**Core Technologies (Inherited from Project):**
- **Python:** Primary programming language (Python 3.10+).
- **OpenAI Gymnasium:** Core library for the environment (`ALE/PongNoFrameskip-v4`).
- **PyTorch:** Deep learning framework for the DQN model.
- **NumPy:** For numerical operations, especially array manipulations for observations, states, and replay buffer.
- **Matplotlib:** For visualization of training progress (e.g., rewards per episode).

**Environment-Specific Technologies & Libraries:**
- **`ale-py`:** The Arcade Learning Environment (ALE) Python interface, required by Gymnasium to run Atari game environments like Pong. This should be part of the project's `requirements.txt`.
- **`opencv-python` (cv2):** Used for image preprocessing:
    - Grayscale conversion (`cv2.cvtColor`).
    - Downsampling/resizing (`cv2.resize`).
    This should also be part of `requirements.txt`.

**Modular Code Organization:**
The implementation has been refactored into a modular structure consisting of:
- **`pong_dqn.py`:** Main entry point with dependency checking and command-line interface
- **`pong_dqn_utils.py`:** Preprocessing, frame stacking, replay buffer, and general utilities
- **`pong_dqn_model.py`:** PyTorch neural network model and agent implementation
- **`pong_dqn_vis.py`:** Visualization and plotting functionality
- **`pong_dqn_train.py`:** Training and evaluation logic

This modular structure improves:
- Code maintainability through better separation of concerns
- Readability by reducing file sizes and focusing each file on specific functionality
- Extensibility for future enhancements

**Key Technical Components:**
- **`preprocess(frame)` function (in `pong_dqn_utils.py`):** Handles frame conversion to grayscale, resizing to 84x84, and normalization to [0,1].
- **`FrameStack` class (in `pong_dqn_utils.py`):** Manages a deque of the last `k` (e.g., 4) preprocessed frames to form the agent's state.
- **`PongDQN(nn.Module)` class (in `pong_dqn_model.py`):** The PyTorch neural network model implementing the CNN architecture (3 conv layers + 2 fc layers).
- **`DQNAgent` class (in `pong_dqn_model.py`):** Manages the policy and target networks, exploration strategy, and learning algorithm.
- **`ReplayBuffer` class (in `pong_dqn_utils.py`):** Stores experiences `(state, action, reward, next_state, done)` and allows for random sampling of minibatches.
- **`plot_training_data` function (in `pong_dqn_vis.py`):** Creates multi-axis visualizations of training metrics (rewards, Q-values, loss).
- **`train_pong_agent` and `evaluate_pong_agent` functions (in `pong_dqn_train.py`):** Implement the main training and evaluation loops.

**Hardware & Performance:**
- **CPU:** Preprocessing and environment interaction will primarily run on the CPU.
- **GPU (PyTorch with MPS on Apple Silicon, or CUDA on NVIDIA):** Neural network training (forward and backward passes) will be accelerated on the GPU if available and PyTorch is configured correctly. The `PongDQN` model and tensors should be moved to the appropriate device (e.g., `torch.device("mps")` or `torch.device("cuda")` if available, otherwise `torch.device("cpu")`).
- **Device Management:** The refactored code centralizes device management in `pong_dqn_model.py`, making it easier to ensure all tensors are on the correct device.

**Configuration & Hyperparameters:**
- Key hyperparameters (learning rate, batch size, buffer size, discount factor, target update frequency, epsilon decay schedule) are defined in `approaches.md` and implemented in the training code.
- Experiment 1 settings (LR: `2.5e-4`, Target Update: `10,000` steps) are consistently applied across the codebase.

**Dependencies (to ensure are in global `requirements.txt`):**
- `gymnasium`
- `ale-py`
- `torch`
- `numpy`
- `matplotlib`
- `opencv-python`

**Development Considerations:**
- **Reproducibility:** Use of fixed random seeds (`random.seed()`, `np.random.seed()`, `torch.manual_seed()`) is crucial for debugging and comparing runs.
- **Checkpointing:** The implementation includes robust checkpoint saving and loading, allowing training to be paused and resumed.
- **Training Insights:** Enhanced logging (average max Q-value and average loss per episode) provides better visibility into learning dynamics.
- **Code Maintenance:** The modular structure makes it easier to update or extend functionality without affecting other components.
