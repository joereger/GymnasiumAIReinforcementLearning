"""
Visualization utility for the trained PPO agent on Pong.
This script loads a trained model and visualizes its gameplay.
"""

import argparse
import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from pong_env_wrappers import make_pong_env, device
from pong_ppo_model import PPOActorCritic

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained PPO agent on Pong")
    parser.add_argument("--model-path", type=str, default="data/pong/models/ppo_pong_final.pt",
                        help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to play")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("--save-video", action="store_true",
                        help="Save a video of the agent's gameplay")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for rendered video")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="Epsilon for random action probability")
    
    return parser.parse_args()

def save_frame(frame, frame_idx, output_dir):
    """Save a single frame to disk."""
    im = Image.fromarray(frame)
    im.save(os.path.join(output_dir, f"frame_{frame_idx:05d}.png"))

def create_video(output_dir, video_path, fps=30):
    """Create a video from saved frames."""
    frames_path = os.path.join(output_dir, "frame_%05d.png")
    os.system(f"ffmpeg -framerate {fps} -i {frames_path} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {video_path}")

def visualize_policy_attention(agent, state, action, save_path=None):
    """
    Visualize where the policy is "looking" by creating a simple saliency map.
    This is a basic version that just shows intermediate activations.
    """
    # We'll use a simple activation visualization from the first convolutional layer
    # In a more sophisticated implementation, we'd compute gradients w.r.t the input
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    
    # Forward pass to get activations
    features = agent.cnn_base[0](state_tensor)  # Get first conv layer activations
    
    # Take the mean across channels and normalize
    attention_map = features.mean(dim=1).squeeze().detach().cpu().numpy()
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # Resize to match input dimensions
    attention_map = cv2.resize(attention_map, (84, 84))
    
    # Create a heatmap
    heatmap = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Get the first channel of the state for visualization
    state_img = state[0]  # First channel of the state
    state_img = (state_img * 255).astype(np.uint8)
    state_img = cv2.cvtColor(state_img, cv2.COLOR_GRAY2BGR)
    
    # Overlay heatmap on the state image
    overlaid = cv2.addWeighted(state_img, 0.7, heatmap, 0.3, 0)
    
    # Add action information
    action_names = ["STAY", "UP", "DOWN"]
    action_text = f"Action: {action_names[action]}"
    cv2.putText(overlaid, action_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if save_path:
        cv2.imwrite(save_path, overlaid)
    
    return overlaid

def visualize_ppo_agent():
    """Visualize trained PPO agent."""
    args = parse_args()
    
    # Create environment using our robust environment creation function
    env = make_pong_env(render_mode="human" if args.render else None, reduced_actions=True)
    
    # Get environment info
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Using device: {device}")
    
    # Create actor-critic agent
    agent = PPOActorCritic(
        input_channels=obs_shape[0],
        action_dim=n_actions
    ).to(device)
    
    # Load model
    if os.path.exists(args.model_path):
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Model not found at {args.model_path}, using random policy")
    
    # Setup for video recording
    if args.save_video:
        output_dir = os.path.join("data", "pong", "videos")
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_idx = 0
    
    # Play episodes
    total_reward = 0
    action_counts = {0: 0, 1: 0, 2: 0}  # Count actions taken
    
    for episode in range(args.episodes):
        state, _ = env.reset(seed=np.random.randint(0, 1000000))
        episode_reward = 0
        episode_length = 0
        done = False
        
        print(f"\nStarting Episode {episode + 1}")
        
        while not done:
            # Choose action
            if np.random.random() < args.epsilon:
                action = env.action_space.sample()
                action_source = "random"
            else:
                state_tensor = torch.FloatTensor(state).to(device)
                with torch.no_grad():
                    action, _, _ = agent.get_action(state_tensor, deterministic=args.deterministic)
                action_source = "deterministic" if args.deterministic else "stochastic"
            
            # Count action
            action_counts[action] += 1
            
            # Visualize policy attention
            if args.save_video and episode_length % 30 == 0:  # Every 30 frames
                viz_img = visualize_policy_attention(
                    agent, 
                    state, 
                    action, 
                    save_path=os.path.join(output_dir, f"attention_ep{episode+1}_step{episode_length}.png")
                )
            
            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Get rendered frame if needed
            if args.save_video and args.render:
                frame = env.render()
                save_frame(frame, frame_idx, frames_dir)
                frame_idx += 1
            
            # Print reward received
            if reward != 0:
                print(f"Step {episode_length}: Action={action} ({action_source}), Reward={reward}")
            
            # Update stats
            episode_reward += reward
            episode_length += 1
            
            # Slow down visualization
            if args.render:
                time.sleep(1 / args.fps)
            
            # Update state
            state = next_state
        
        # Episode summary
        print(f"Episode {episode + 1} finished: Reward={episode_reward}, Length={episode_length}")
        total_reward += episode_reward
    
    # Overall summary
    avg_reward = total_reward / args.episodes
    print(f"\nAverage reward over {args.episodes} episodes: {avg_reward:.2f}")
    print(f"Action distribution: {action_counts}")
    
    # Create video from frames
    if args.save_video and args.render:
        video_path = os.path.join(output_dir, f"ppo_pong_{int(time.time())}.mp4")
        create_video(frames_dir, video_path, fps=args.fps)
        print(f"Video saved to {video_path}")
    
    # Clean up
    env.close()

if __name__ == "__main__":
    visualize_ppo_agent()
