# Technical Context: Freeway

This document details the technical stack, dependencies, and environment setup for the Freeway solution.

## Core Dependencies

The Freeway implementation uses the following key technologies and libraries:

1. **PyTorch**: Deep learning framework used for implementing the neural network
   - Version: As specified in `requirements.txt` (torch==2.7.0)
   - Used for building the DQN model, computing loss, and optimizing the network

2. **Gymnasium**: Reinforcement learning environment framework
   - Version: As specified in `requirements.txt` (gymnasium==1.1.1)
   - Atari extension: Requires `gymnasium[atari,accept-rom-license]`
   - Provides the Freeway environment through the Arcade Learning Environment (ALE)

3. **OpenCV (cv2)**: Image processing library
   - Used for preprocessing: grayscale conversion, resizing
   - Required for the observation preprocessing pipeline

4. **NumPy**: Numerical computing library
   - Used for array manipulation throughout the implementation
   - Required for batch processing and data handling

5. **Matplotlib**: Visualization library
   - Used for plotting training progress metrics
   - Generates reward curves, episode length plots, and Q-value trends

## Installation Requirements

The implementation requires the following installation steps:

1. **Base Environment Setup**:
   ```bash
   # From project root
   pip install -r requirements.txt
   ```

2. **Atari-Specific Dependencies**:
   For zsh shell (Mac/Linux):
   ```bash
   # The brackets need quoting in zsh
   pip install ale-py
   pip install 'gymnasium[accept-rom-license]'
   ```
   
   For bash/other shells:
   ```bash
   pip install ale-py
   pip install gymnasium[accept-rom-license]
   ```

3. **ROM Availability Check**:
   The code includes a verification step to ensure Atari and Freeway are properly set up:
   ```python
   # Import and register Atari environments
   try:
       import ale_py
       gym.register_envs(ale_py)
       atari_available = True
   except ImportError:
       atari_available = False
       print("Warning: ale_py not found, Atari environments will not be available")
       print("Install with: pip install ale-py")
       
   # Check for Freeway ROM
   try:
       env = gym.make("ALE/Freeway-v5")
       env.close()
       print("Freeway ROM successfully loaded.")
   except Exception as e:
       print(f"Error loading Freeway ROM: {e}")
   ```

## Hardware Considerations

- **GPU Acceleration**: The implementation detects and uses CUDA if available
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
  
- **Memory Requirements**:
  - Replay Buffer: Configured to store 10,000 experiences
  - Each experience contains multiple stacked frames, requiring substantial memory
  - Recommended: At least 8GB RAM for comfortable training

- **Storage Requirements**:
  - Models are saved in the `data/freeway/` directory
  - Each saved model is approximately 5-20MB
  - Training plots and checkpoints require additional storage

## Development Environment

The code is designed to run in a Python environment with the following characteristics:

- **Python Version**: 3.8+ recommended
- **Virtual Environment**: Recommended to use a dedicated virtual environment
- **File Structure**:
  ```
  /Gynasium                  # Project root
  ├── freeway/               # Freeway implementation
  │   └── freeway.py         # Main implementation file
  ├── data/                  # Data directory
  │   └── freeway/           # Freeway-specific data
  │       ├── freeway_dqn.pth            # Model checkpoint
  │       ├── freeway_dqn_best.pth       # Best model
  │       └── training_progress.png      # Training curves
  └── memory-bank/          # Documentation
      └── environments/
          └── freeway/       # Freeway documentation
  ```

## Technical Constraints

1. **ALE Integration Challenges**:
   - The RAM-based position tracking (`ale.getRAM()`) requires working with ALE's memory layout
   - Position tracking at RAM address 0x94 might need adjustment for different ALE versions

2. **Frame Preprocessing Pipeline**:
   - The standard 84x84 preprocessing has computation costs
   - Trade-off between image size, information preservation, and computation speed

3. **PyTorch GPU Memory Management**:
   - Tensors moved to GPU with `.to(device)`
   - Gradients explicitly clipped to prevent training instability
   - Explicit `.detach()` calls used where appropriate to manage computational graph

## Tools and Usage Patterns

1. **Training/Evaluation Control Flow**:
   - Interactive command-line interface for choosing mode
   - Options for enabling/disabling rendering
   - Checkpoint loading capability for continued training

2. **Visualization Approach**:
   - Real-time rendering via Gymnasium/ALE
   - Post-hoc analysis through saved matplotlib plots
   - Single-line logging for compact training progress tracking

3. **Data Management**:
   - Environment-specific data directory for model persistence
   - Automatic creation of required directories
   - Best-model saving based on performance metrics
