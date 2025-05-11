"""
Pong Double DQN Training and Evaluation (Experiment 2)

Main file for running Double DQN training and evaluation on the Pong Atari environment.
This implementation addresses the Q-value collapse observed in the vanilla DQN approach.

Key Improvements in Experiment 2:
- Double DQN algorithm to reduce overestimation bias
- Lower learning rate (1e-5 vs 2.5e-4)
- More frequent target network updates (1,000 vs 10,000 steps)
- Larger replay buffer (500K vs 100K)
- Slower epsilon decay (2M vs 1M frames)
"""

import gymnasium as gym
import os
import sys
import ale_py

# Import module functions
from pong_double_dqn_train import train_double_dqn_agent, evaluate_double_dqn_agent

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        
        # Check for CUDA/MPS availability
        from pong_double_dqn_model import device
        print(f"Using device: {device}")
        
        # Check for ALE support
        if not 'ale_py' in globals() or not hasattr(ale_py, '__version__'):
            print("WARNING: ale_py not properly imported or available. Please install with:")
            print("pip install ale-py")
            return False
            
        # Try registering ALE environments
        gym.register_envs(ale_py)
        
        # Check if Pong ROM is available
        try:
            temp_env_check = gym.make("PongNoFrameskip-v4")
            temp_env_check.close()
            print("Pong ROM successfully loaded via ale_py.")
        except Exception as e:
            print(f"Error loading PongNoFrameskip-v4: {e}")
            print("This might be due to missing ROM files if ale_py is installed but ROMs are not.")
            print("If needed, you might need to install ROMs, e.g., `pip install gymnasium[accept-rom-license]` (use quotes in zsh).")
            return False
            
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all required packages.")
        return False

def setup_paths():
    """Setup paths for models and data storage."""
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pong")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    MODEL_PATH = os.path.join(DATA_DIR, "pong_double_dqn_model.pth")
    BEST_MODEL_PATH = os.path.join(DATA_DIR, "pong_double_dqn_best_model.pth")
    
    return MODEL_PATH, BEST_MODEL_PATH, DATA_DIR

def main():
    """Main entry point for training and evaluation."""
    # Check if we can run
    if not check_dependencies():
        sys.exit(1)
        
    # Setup paths
    MODEL_PATH, BEST_MODEL_PATH, _ = setup_paths()
    
    print("=" * 80)
    print("DOUBLE DQN EXPERIMENT 2")
    print("Key differences from original implementation:")
    print("1. Double DQN algorithm: Uses online network to SELECT and target network to EVALUATE actions")
    print("2. Learning rate: 1e-5 (reduced from 2.5e-4)")
    print("3. Target network updates: Every 1,000 steps (more frequent than 10,000)")
    print("4. Replay buffer size: 500K (increased from 100K)")
    print("5. Epsilon decay: Over 2M frames (slower than 1M)")
    print("=" * 80)
    
    # Main interaction loop
    while True:
        choice = input("Do you want to [t]rain or [e]valuate? ").lower()
        
        if choice in ['t', 'train']:
            render_train_choice = input("Enable human rendering during training? [y/n]: ").lower()
            do_human_render_train = render_train_choice in ['y', 'yes']
            
            load_choice = input("Load existing checkpoint to continue training? [y/n]: ").lower()
            load_checkpoint_train = load_choice in ['y', 'yes']
            
            print("\nStarting Double DQN training...")
            train_double_dqn_agent(human_render_during_training=do_human_render_train, 
                                  load_checkpoint_flag=load_checkpoint_train)
            break
            
        elif choice in ['e', 'evaluate']:
            render_eval_choice = input("Enable human rendering during evaluation? [y/n]: ").lower()
            do_human_render_eval = render_eval_choice in ['y', 'yes']
            
            model_to_eval = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
            if not os.path.exists(model_to_eval):
                print(f"No model found at {model_to_eval} or {BEST_MODEL_PATH}. Please train a model first.")
                break 

            print(f"\nEvaluating Double DQN agent using model: {model_to_eval}...")
            evaluate_double_dqn_agent("PongNoFrameskip-v4", model_to_eval, 
                                     num_episodes=10, human_render=do_human_render_eval)
            break
            
        else:
            print("Invalid choice. Please enter 't' for train or 'e' for evaluate.")

if __name__ == "__main__":
    main()
