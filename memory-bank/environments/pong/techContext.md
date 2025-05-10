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

**Key Technical Components (as defined in `systemPatterns.md`):**
- **`preprocess(frame)` function:** Handles frame conversion to grayscale, resizing to 84x84, and normalization to [0,1].
- **`FrameStack` class:** Manages a deque of the last `k` (e.g., 4) preprocessed frames to form the agent's state.
- **`PongDQN(nn.Module)` class:** The PyTorch neural network model implementing the CNN architecture (3 conv layers + 2 fc layers).
- **`ReplayBuffer` class:** Stores experiences `(state, action, reward, next_state, done)` and allows for random sampling of minibatches.

**Hardware & Performance:**
- **CPU:** Preprocessing and environment interaction will primarily run on the CPU.
- **GPU (PyTorch with MPS on Apple Silicon, or CUDA on NVIDIA):** Neural network training (forward and backward passes) will be accelerated on the GPU if available and PyTorch is configured correctly. The `PongDQN` model and tensors should be moved to the appropriate device (e.g., `torch.device("mps")` or `torch.device("cuda")` if available, otherwise `torch.device("cpu")`).

**Configuration & Hyperparameters:**
- Key hyperparameters (learning rate, batch size, buffer size, discount factor, target update frequency, epsilon decay schedule) are defined in `approaches.md` and will be implemented in the `pong_dqn.py` script.

**Dependencies (to ensure are in global `requirements.txt`):**
- `gymnasium`
- `ale-py`
- `torch`
- `numpy`
- `matplotlib`
- `opencv-python`

**Development Considerations:**
- **Reproducibility:** Use of fixed random seeds (`random.seed()`, `np.random.seed()`, `torch.manual_seed()`) is crucial for debugging and comparing runs.
- **Device Management:** Explicitly manage tensor device placement (e.g., `.to(device)`) for model parameters and data.
