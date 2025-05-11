"""
Pong DQN Training and Evaluation

Main file for running DQN training and evaluation on the Pong Atari environment.
This is the entry point to the program that imports functionality from the split modules.

Current Experiment 1 Parameters:
- Learning Rate: 2.5e-4
- Target Network Update Frequency: 10,000 steps

Other important features:
- Reward clipping to [-1, 1]
- 50k step replay buffer warmup
- Gradient clipping to 1.0
- Enhanced logging (avg_loss, avg_max_q)
- Multi-axis plotting showing all metrics
"""

import gymnasium as gym
import os
import sys
import ale_py

# Import module functions
from pong_dqn_train import train_pong_agent, evaluate_pong_agent

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        
        # Check for CUDA/MPS availability
        from pong_dqn_model import device
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
    
    MODEL_PATH = os.path.join(DATA_DIR, "pong_dqn_model.pth")
    BEST_MODEL_PATH = os.path.join(DATA_DIR, "pong_dqn_best_model.pth")
    
    return MODEL_PATH, BEST_MODEL_PATH, DATA_DIR

def main():
    """Main entry point for training and evaluation."""
    # Check if we can run
    if not check_dependencies():
        sys.exit(1)
        
    # Setup paths
    MODEL_PATH, BEST_MODEL_PATH, _ = setup_paths()
    
    # Main interaction loop
    while True:
        choice = input("Do you want to [t]rain or [e]valuate? ").lower()
        
        if choice in ['t', 'train']:
            render_train_choice = input("Enable human rendering during training? [y/n]: ").lower()
            do_human_render_train = render_train_choice in ['y', 'yes']
            
            load_choice = input("Load existing checkpoint to continue training? [y/n]: ").lower()
            load_checkpoint_train = load_choice in ['y', 'yes']
            
            print("\nStarting training...")
            train_pong_agent(human_render_during_training=do_human_render_train, 
                            load_checkpoint_flag=load_checkpoint_train)
            break
            
        elif choice in ['e', 'evaluate']:
            render_eval_choice = input("Enable human rendering during evaluation? [y/n]: ").lower()
            do_human_render_eval = render_eval_choice in ['y', 'yes']
            
            model_to_eval = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
            if not os.path.exists(model_to_eval):
                print(f"No model found at {model_to_eval} or {BEST_MODEL_PATH}. Please train a model first.")
                break 

            print(f"\nEvaluating agent using model: {model_to_eval}...")
            evaluate_pong_agent("PongNoFrameskip-v4", model_to_eval, 
                              num_episodes=10, human_render=do_human_render_eval)
            break
            
        else:
            print("Invalid choice. Please enter 't' for train or 'e' for evaluate.")

if __name__ == "__main__":
    main()
