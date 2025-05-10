# Tech Context: Gymnasium Environment Solutions

**Core Technologies:**
- **Python:** The primary programming language for all development. Python 3.10+ is used.
- **OpenAI Gymnasium:** The core library providing the suite of reinforcement learning environments. Using Gymnasium 1.1.1.
- **PyTorch:** Deep learning framework used for implementing neural network-based agents (version 2.7.0).
- **Reinforcement Learning Algorithms:** Custom implementations of various algorithms:
  - DQN (Deep Q-Network) and variants (Double DQN)
  - A3C (Asynchronous Advantage Actor-Critic)
  - Genetic Algorithms
- **NumPy:** For numerical operations, especially array manipulations for observations and actions.
- **Matplotlib:** For visualization of training progress, rewards, and agent performance.
- **OpenCV (cv2):** Used for image preprocessing in pixel-based environments like Atari games.

**Development Setup:**
- **Operating System:** macOS on Apple Silicon (M1).
- **IDE/Editor:** VSCode.
- **Virtual Environment:** Managed via `GymnasiumVENV/` using venv.
- **Version Control:** Git.
- **Hardware Acceleration:** PyTorch with MPS (Metal Performance Shaders) support for Apple Silicon GPUs.

**Technical Constraints:**
- Solutions should be runnable within the provided Gymnasium environments.
- Compatibility with Apple Silicon hardware is prioritized to leverage MPS acceleration.
- Different environments require different approaches - some benefit from deep learning, others from simpler algorithms.

**Dependencies:**
- Dependencies are managed via `requirements.txt`
- Core dependencies include:
  - `gymnasium==1.1.1`
  - `torch==2.7.0`
  - `numpy==1.25.2`
  - `matplotlib==3.10.3`
  - `box2d-py==2.3.8` (for BipedalWalker, LunarLander)
  - `ale-py>=0.8.0` (for Atari environments like Freeway)
  - `opencv-python>=4.5.0` (for image preprocessing)

**Tool Usage Patterns:**
- **Cline:** Used as an AI software engineer, relying on the Memory Bank for context.
- **VSCode:** Primary development environment.
- **Terminal (within VSCode or standalone):** For running Python scripts, managing virtual environments, and Git operations.
- **PyTorch with MPS:** For leveraging Apple Silicon GPU acceleration where possible, especially for training deep neural networks.
- **Environment-Specific Tools:**
  - Atari: Uses `ale-py` and frame preprocessing tools
  - Box2D: Uses `box2d-py` for physics simulation
