"""
Pong Proximal Policy Optimization (PPO) Training and Evaluation (Experiment 3)

Main file for running PPO training and evaluation on the Pong Atari environment.
This implementation addresses the learning issues observed in DQN and Double DQN approaches.

Key Improvements in Experiment 3:
- Shifts from value-based (DQN) to policy gradient (PPO) methods
- Uses Actor-Critic architecture for better policy learning
- Implements PPO's clipped surrogate objective for stable updates
- Leverages Generalized Advantage Estimation (GAE) 
- Uses stochastic policy for better exploration
"""

import gymnasium as gym
import os
import sys
import ale_py

# Import module functions
from pong_ppo_train import train_ppo_agent, evaluate_ppo_agent
from pong_ppo_model import PPOAgent, device

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
        
        # Check for GPU/MPS availability
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
    
    MODEL_PATH = os.path.join(DATA_DIR, "pong_ppo_model.pth")
    BEST_MODEL_PATH = os.path.join(DATA_DIR, "pong_ppo_best_model.pth")
    
    return MODEL_PATH, BEST_MODEL_PATH, DATA_DIR

def print_ppo_info():
    """Print information about the PPO approach and why it might work better than DQN."""
    print("=" * 80)
    print("PPO EXPERIMENT 3")
    print("After issues with DQN and Double DQN implementations, we're switching to PPO:")
    print("1. Policy Gradient vs Value Based: PPO directly optimizes the policy instead of estimating Q-values")
    print("2. Actor-Critic Architecture: Combines policy optimization with value function estimation")
    print("3. Clipped Surrogate Objective: Limits policy updates for more stable learning")
    print("4. Generalized Advantage Estimation: Better credit assignment for sparse rewards")
    print("5. Stochastic Policy: Naturally handles exploration without epsilon-greedy")
    print("=" * 80)
    print("Expected Benefits:")
    print("- No Q-value collapse issues that we saw in DQN/Double DQN")
    print("- Better handling of sparse rewards in Pong (only receive rewards when scoring)")
    print("- More stable and consistent learning progress")
    print("- Policy gradient methods often more sample efficient for Atari games")
    print("=" * 80)

def main():
    """Main entry point for training and evaluation."""
    # Check if we can run
    if not check_dependencies():
        sys.exit(1)
        
    # Setup paths
    MODEL_PATH, BEST_MODEL_PATH, _ = setup_paths()
    
    # Print info about PPO approach
    print_ppo_info()
    
    # Main interaction loop
    while True:
        choice = input("Do you want to [t]rain or [e]valuate? ").lower()
        
        if choice in ['t', 'train']:
            # Training options
            render_train_choice = input("Enable human rendering during training? [y/n]: ").lower()
            do_human_render_train = render_train_choice in ['y', 'yes']
            
            load_choice = input("Load existing checkpoint to continue training? [y/n]: ").lower()
            load_checkpoint_train = load_choice in ['y', 'yes']
            
            # Advanced hyperparameter options
            advanced_choice = input("Configure advanced hyperparameters? [y/n]: ").lower()
            if advanced_choice in ['y', 'yes']:
                try:
                    rollout_length = int(input("Rollout length (default 128): ") or "128")
                    lr = float(input("Learning rate (default 3e-4): ") or "0.0003")
                    clip_param = float(input("PPO clip parameter (default 0.2): ") or "0.2")
                    entropy_coef = float(input("Entropy coefficient (default 0.01): ") or "0.01")
                    ppo_epochs = int(input("PPO epochs per update (default 4): ") or "4")
                    
                    print("\nStarting PPO training with custom hyperparameters...")
                    train_ppo_agent(
                        rollout_length=rollout_length,
                        lr=lr,
                        clip_param=clip_param,
                        entropy_coef=entropy_coef,
                        ppo_epochs=ppo_epochs,
                        human_render_during_training=do_human_render_train,
                        load_checkpoint_flag=load_checkpoint_train
                    )
                except ValueError as e:
                    print(f"Invalid input: {e}. Using default values.")
                    train_ppo_agent(
                        human_render_during_training=do_human_render_train,
                        load_checkpoint_flag=load_checkpoint_train
                    )
            else:
                print("\nStarting PPO training with default hyperparameters...")
                train_ppo_agent(
                    human_render_during_training=do_human_render_train,
                    load_checkpoint_flag=load_checkpoint_train
                )
            break
            
        elif choice in ['e', 'evaluate']:
            render_eval_choice = input("Enable human rendering during evaluation? [y/n]: ").lower()
            do_human_render_eval = render_eval_choice in ['y', 'yes']
            
            # Check if model exists
            model_to_eval = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
            if not os.path.exists(model_to_eval):
                print(f"No model found at {model_to_eval} or {BEST_MODEL_PATH}. Please train a model first.")
                continue

            # Create agent for evaluation
            from pong_ppo_utils import FrameStack
            import gymnasium as gym
            
            # Get state shape for agent creation
            env_temp = gym.make("PongNoFrameskip-v4")
            obs_temp, _ = env_temp.reset()
            frame_stacker_temp = FrameStack(k=4)
            state_temp = frame_stacker_temp.reset(obs_temp)
            STATE_SHAPE = state_temp.shape
            ACTION_SPACE = env_temp.action_space.n
            env_temp.close()
            
            # Create agent and load model
            agent = PPOAgent(
                state_shape=STATE_SHAPE,
                action_space=ACTION_SPACE
            )
            agent.load(model_to_eval)
            
            # Get number of episodes
            try:
                num_episodes = int(input("Number of evaluation episodes (default 10): ") or "10")
            except ValueError:
                num_episodes = 10
                print("Invalid input. Using default value of 10 episodes.")

            print(f"\nEvaluating PPO agent using model: {model_to_eval}...")
            evaluate_ppo_agent(
                "PongNoFrameskip-v4", 
                agent, 
                num_episodes=num_episodes, 
                human_render=do_human_render_eval
            )
            break
            
        else:
            print("Invalid choice. Please enter 't' for train or 'e' for evaluate.")

if __name__ == "__main__":
    main()
