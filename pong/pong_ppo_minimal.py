"""
Main script for training a minimal PPO implementation on Pong with detailed diagnostics.
"""

import os
import argparse
import torch
import numpy as np
import glob
import datetime
import json
from pathlib import Path

# Import from our modular implementation
from pong_ppo_minimal_env import make_pong_env, device
from pong_ppo_minimal_model import DiagnosticActorCritic
from pong_ppo_minimal_ppo import DiagnosticPPO
from pong_ppo_minimal_train import (
    train, evaluate_agent, visualize_training_progress, collect_episodes,
    global_episode_count, global_step_count
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO on Pong with detailed diagnostics")
    
    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, 
                        help="Total timesteps for training")
    parser.add_argument("--rollout_steps", type=int, default=4096, 
                        help="Number of steps to collect per rollout (larger is more stable)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, 
                        help="Lower learning rate (1e-4) to prevent activation collapse")
    parser.add_argument("--gamma", type=float, default=0.99, 
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, 
                        help="GAE lambda parameter")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, 
                        help="PPO clip parameter (smaller value for more stable learning)")
    parser.add_argument("--entropy_coef", type=float, default=0.01, 
                        help="Entropy coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5, 
                        help="Value loss coefficient")
    
    # Evaluation parameters
    parser.add_argument("--eval_freq", type=int, default=10, 
                        help="Evaluation frequency (in updates)")
    parser.add_argument("--eval_episodes", type=int, default=5, 
                        help="Number of episodes for evaluation")
    
    # Saving and logging
    parser.add_argument("--save_freq", type=int, default=50, 
                        help="Model saving frequency (in updates)")
    parser.add_argument("--log_freq", type=int, default=1, 
                        help="Logging frequency (in episodes)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed")
    parser.add_argument("--diagnostic_mode", action="store_true", 
                        help="Enable detailed diagnostics")
    parser.add_argument("--render", action="store_true", 
                        help="Render the environment during evaluation")
    parser.add_argument("--eval_only", action="store_true", 
                        help="Only run evaluation on a saved model")
    parser.add_argument("--model_path", type=str, default="data/pong/models/best_model.pt", 
                        help="Path to load model for evaluation")
    
    return parser.parse_args()

def list_available_models():
    """List all available model files in the models directory."""
    models_dir = "data/pong/models"
    
    # Check for model files
    model_files = glob.glob(f"{models_dir}/*.pt")
    checkpoint_files = [f for f in model_files if "checkpoint" in f]
    other_files = [f for f in model_files if "checkpoint" not in f]
    
    # Sort checkpoint files by update number
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else 0)
    
    all_files = other_files + checkpoint_files
    
    if not all_files:
        print("No saved models found.")
        return None
    
    print("\nAvailable model files:")
    for i, model_file in enumerate(all_files):
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # Size in MB
        file_date = os.path.getmtime(model_file)
        date_str = datetime.datetime.fromtimestamp(file_date).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{i+1}] {Path(model_file).name} ({file_size:.2f} MB, {date_str})")
    
    return all_files

def gather_user_preferences(args):
    """
    Gather all user preferences at the beginning to avoid interruptions during training.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with user preferences
    """
    print("\n== Training Configuration ==")
    
    # 1. Visualization preferences
    render_response = input("Would you like to visualize the environment during training? (y/n): ").lower()
    render_env = render_response.startswith('y')
    
    # 2. Model loading preferences
    print("\n== Model Loading Options ==")
    load_model_response = input("Would you like to load a pre-trained model? (y/n): ").lower()
    
    model_path = None
    optimizer_state = None
    start_timesteps = 0
    start_updates = 0
    start_episodes = 0
    loaded_episode_records = {}
    
    if load_model_response.startswith('y'):
        # List available models
        all_model_files = list_available_models()
        
        if all_model_files:
            valid_selection = False
            while not valid_selection:
                model_selection = input("\nEnter the number of the model to load (or press Enter to start fresh): ")
                
                if not model_selection.strip():
                    print("Starting with a fresh model.")
                    break
                
                try:
                    model_idx = int(model_selection) - 1
                    if 0 <= model_idx < len(all_model_files):
                        model_path = all_model_files[model_idx]
                        print(f"Loading model from {model_path}...")
                        valid_selection = True
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
    
    return {
        'render_env': render_env,
        'model_path': model_path,
        'optimizer_state': optimizer_state,
        'start_timesteps': start_timesteps,
        'start_updates': start_updates,
        'start_episodes': start_episodes,
        'loaded_episode_records': loaded_episode_records
    }

def main():
    """Main function to setup and run PPO on Pong."""
    args = parse_args()
    
    # Set up directories
    os.makedirs("data/pong", exist_ok=True)
    os.makedirs("data/pong/diagnostics", exist_ok=True)
    os.makedirs("data/pong/models", exist_ok=True)
    
    # Set random seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.manual_seed(args.seed)
    
    # Gather all user preferences at the beginning
    preferences = gather_user_preferences(args)
    render_env = preferences['render_env']
    model_path = preferences['model_path']
    
    # Create environments with user-specified rendering
    print("\nCreating environments...")
    env = make_pong_env(render_mode="human" if render_env else None, seed=args.seed)
    # Always render eval environment if the user wants to see it
    eval_env = make_pong_env(render_mode="human" if render_env or args.render else None, seed=args.seed + 1000)
    
    # Get environment information
    input_channels = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create actor-critic network
    print("\nCreating actor-critic network...")
    agent = DiagnosticActorCritic(
        input_channels=input_channels,
        action_dim=action_dim
    ).to(device)
    
    # Variables for training state
    optimizer_state = None
    start_timesteps = 0
    start_updates = 0
    start_episodes = 0
    loaded_episode_records = {}
    
    # Load model if specified
    if model_path:
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if it's a checkpoint with optimizer state
        if isinstance(checkpoint, dict) and 'agent' in checkpoint:
            agent.load_state_dict(checkpoint['agent'])
            optimizer_state = checkpoint.get('optimizer', None)
            start_updates = checkpoint.get('update', 0)
            start_timesteps = checkpoint.get('timesteps', 0)
            start_episodes = checkpoint.get('episode', 0)
            
            # Restore diagnostic counter for continuous activation visualization
            if 'diagnostic_counter' in checkpoint:
                diagnostic_counter = checkpoint['diagnostic_counter']
                agent.set_diagnostic_counter(diagnostic_counter)
                print(f"Restored diagnostic counter: {diagnostic_counter}")
            else:
                print("No diagnostic counter found in checkpoint, using default (0)")
            
            print(f"Resuming from update {start_updates}, episode {start_episodes}, timestep {start_timesteps}")
            
            # Try to load existing training data for continuity
            training_data_path = "data/pong/training_data.json"
            if os.path.exists(training_data_path):
                try:
                    with open(training_data_path, "r") as f:
                        loaded_episode_records = json.load(f)
                        print(f"Loaded existing training history with {len(loaded_episode_records)} episodes.")
                except Exception as e:
                    print(f"Error loading training data: {e}")
        else:
            # Simple model state dict
            agent.load_state_dict(checkpoint)
            print("Loaded model weights only (no training state).")
    
    # Create PPO algorithm
    print("Creating PPO algorithm...")
    ppo = DiagnosticPPO(
        actor_critic=agent,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        diagnostic_mode=args.diagnostic_mode
    )
    
    # Evaluation only mode
    if args.eval_only:
        print(f"Loading model from {args.model_path}...")
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
        
        print("Running evaluation...")
        avg_reward, avg_length, episode_rewards, action_counts = evaluate_agent(
            env=eval_env,
            agent=agent,
            num_episodes=args.eval_episodes,
            render=args.render
        )
        
        print(f"Evaluation results:")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Average length: {avg_length:.2f}")
        print(f"  All rewards: {episode_rewards}")
        
        return
    
    # Training mode
    print("Starting training...")
    
    # If we're loading from a checkpoint, restore optimizer state
    if optimizer_state is not None:
        ppo.optimizer.load_state_dict(optimizer_state)
        print(f"Restored optimizer state from checkpoint.")
    
    # Initialize the global counters for training resumption
    global global_episode_count, global_step_count
    global_episode_count = start_episodes
    global_step_count = start_timesteps
    
    agent, episode_records = train(
        env=env,
        eval_env=eval_env,
        agent=agent,
        ppo=ppo,
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        optimizer_state=optimizer_state,
        start_timesteps=start_timesteps,
        start_updates=start_updates,
        start_episodes=start_episodes,
        loaded_episode_records=loaded_episode_records
    )
    
    # Final visualization
    visualize_training_progress(episode_records, "data/pong/pong_ppo_minimal_progress.png")
    
    print("Training completed! Final model saved to data/pong/models/final_model.pt")

if __name__ == "__main__":
    main()
