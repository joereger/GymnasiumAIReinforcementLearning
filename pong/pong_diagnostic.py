"""
Pong Diagnostic Tool

This script analyzes potential structural issues in the Pong environment implementation
that might be causing learning problems across different algorithms.

It:
1. Implements proper Atari wrappers based on standard literature
2. Visualizes preprocessed frames to verify what the agent sees
3. Tests simple policies to verify environment interaction
4. Tracks statistics like reward distribution and state values
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
import time
import argparse
import ale_py
from collections import deque, Counter
from PIL import Image

# --- Environment Wrappers (Standard in Atari literature) ---
class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        info = None
        for _ in range(noops):
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    Pong requires pressing FIRE to start the game.
    """
    def __init__(self, env):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # RIGHT (to start game)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """
    def __init__(self, env):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame and max over the last 2 frames to account for
    flickering in some Atari games.
    """
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        terminated = truncated = False
        info = None
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {-1, 0, 1} as done in the original DQN paper.
    """
    def __init__(self, env):
        super(ClipRewardEnv, self).__init__(env)
    
    def reward(self, reward):
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in the Nature paper and later work.
    """
    def __init__(self, env, width=84, height=84, grayscale=True):
        super(WarpFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            num_colors = 1
        else:
            num_colors = 3
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, num_colors),
            dtype=np.uint8,
        )
        original_space = self.observation_space
        self.observation_space = new_space
        # Save the original and new spaces to debug or visualize differences
        self.original_space = original_space

    def observation(self, obs):
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(
            obs, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        if self.grayscale:
            obs = np.expand_dims(obs, -1)
        return obs

class FrameStack(gym.Wrapper):
    """
    Stack k last frames.
    Returns lazy array, which is much more memory efficient.
    """
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=2)

class ScaledFloatFrame(gym.ObservationWrapper):
    """Normalize observations to [0, 1]."""
    def __init__(self, env):
        super(ScaledFloatFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=1, 
            shape=env.observation_space.shape, 
            dtype=np.float32
        )

    def observation(self, observation):
        # Careful! This undoes the memory optimization of FrameStack
        return np.array(observation).astype(np.float32) / 255.0

# --- Helper Functions ---
def make_atari_env(env_id, render_mode=None, max_episode_steps=None):
    """Create a properly wrapped Atari environment."""
    env = gym.make(env_id, render_mode=render_mode, repeat_action_probability=0.0, full_action_space=False)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    # Apply wrappers in the standard order
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)

    return env

def visualize_preprocessed_frames(frames, orig_frame=None, title="Preprocessed Frames"):
    """Visualize the preprocessed frames to debug what the agent sees."""
    plt.figure(figsize=(15, 8))
    
    # Show original frame if available
    if orig_frame is not None:
        plt.subplot(1, 5, 1)
        plt.imshow(orig_frame)
        plt.title("Original Frame")
        plt.axis('off')
    
    # Show the 4 stacked frames
    for i in range(4):
        plt.subplot(1, 5, i+2)
        plt.imshow(frames[:, :, i], cmap='gray')
        plt.title(f"Frame {i+1}")
        plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the visualization
    os.makedirs('data/pong/diagnostics', exist_ok=True)
    plt.savefig(f'data/pong/diagnostics/preprocessed_frames_{int(time.time())}.png')
    plt.close()

def run_random_policy(env, num_episodes=5, visualize_every=100):
    """Run a random policy to test environment interactions."""
    print("\n--- TESTING RANDOM POLICY ---")
    total_rewards = []
    action_counts = Counter()
    step_counts = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Sample random action
            action = env.action_space.sample()
            action_counts[action] += 1
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Visualize occasionally
            if steps % visualize_every == 0:
                visualize_preprocessed_frames(
                    next_obs, 
                    title=f"Random Policy - Episode {episode+1}, Step {steps}, Action {action}, Reward {reward}"
                )
            
            # Debug info
            if reward != 0:
                print(f"Episode {episode+1}, Step {steps}: Action {action} received reward {reward}")
        
        total_rewards.append(episode_reward)
        step_counts.append(steps)
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {steps}")
    
    print(f"\nRandom Policy Stats:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Steps: {np.mean(step_counts):.2f}")
    print(f"Action Distribution: {dict(action_counts)}")
    return total_rewards, action_counts, step_counts

def run_simple_policy(env, policy_type="up", num_episodes=5, visualize_every=100):
    """Run a simple deterministic policy (always up, down, or alternating)."""
    print(f"\n--- TESTING {policy_type.upper()} POLICY ---")
    total_rewards = []
    step_counts = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            # Determine action based on policy type
            if policy_type == "up":
                action = 2  # UP action
            elif policy_type == "down":
                action = 3  # DOWN action
            elif policy_type == "alternating":
                action = 2 if steps % 2 == 0 else 3  # Alternate UP and DOWN
            else:
                action = 0  # NOOP
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Visualize occasionally
            if steps % visualize_every == 0:
                visualize_preprocessed_frames(
                    next_obs, 
                    title=f"{policy_type.capitalize()} Policy - Episode {episode+1}, Step {steps}, Reward {reward}"
                )
            
            # Debug info
            if reward != 0:
                print(f"Episode {episode+1}, Step {steps}: Action {action} received reward {reward}")
        
        total_rewards.append(episode_reward)
        step_counts.append(steps)
        print(f"Episode {episode+1}: Reward = {episode_reward}, Steps = {steps}")
    
    print(f"\n{policy_type.capitalize()} Policy Stats:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    print(f"Average Steps: {np.mean(step_counts):.2f}")
    return total_rewards, step_counts

def analyze_state_values(env, num_steps=1000):
    """Analyze state values to see if the preprocessing pipeline is working correctly."""
    print("\n--- ANALYZING STATE VALUES ---")
    obs, info = env.reset()
    
    # Collect state statistics
    pixel_means = []
    pixel_stds = []
    pixel_mins = []
    pixel_maxs = []
    
    for step in range(num_steps):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Analyze frame statistics
        pixel_means.append(np.mean(next_obs))
        pixel_stds.append(np.std(next_obs))
        pixel_mins.append(np.min(next_obs))
        pixel_maxs.append(np.max(next_obs))
        
        # Reset if done
        if terminated or truncated:
            obs, info = env.reset()
    
    # Report statistics
    print(f"Pixel Mean: {np.mean(pixel_means):.6f}")
    print(f"Pixel Std: {np.mean(pixel_stds):.6f}")
    print(f"Pixel Min: {np.min(pixel_mins):.6f}")
    print(f"Pixel Max: {np.max(pixel_maxs):.6f}")
    
    # Create histogram of pixel values
    plt.figure(figsize=(10, 6))
    plt.hist(pixel_means, bins=30)
    plt.title("Distribution of Mean Pixel Values")
    plt.xlabel("Mean Pixel Value")
    plt.ylabel("Frequency")
    os.makedirs('data/pong/diagnostics', exist_ok=True)
    plt.savefig(f'data/pong/diagnostics/pixel_value_distribution.png')
    plt.close()

def main():
    """Main diagnostic function"""
    parser = argparse.ArgumentParser(description="Pong Environment Diagnostics")
    parser.add_argument("--render", action="store_true", help="Render the environment visually")
    args = parser.parse_args()
    
    render_mode = "human" if args.render else None
    
    print("=" * 80)
    print("PONG ENVIRONMENT DIAGNOSTIC TOOL")
    print("=" * 80)
    print("\nThis script will diagnose potential issues in the Pong environment implementation.")
    
    # Print action meanings for understanding
    env_temp = gym.make("PongNoFrameskip-v4")
    action_meanings = env_temp.unwrapped.get_action_meanings()
    env_temp.close()
    
    print("\nAction Meanings in Pong:")
    for i, meaning in enumerate(action_meanings):
        print(f"Action {i}: {meaning}")
    
    # Create properly wrapped environment
    print("\nCreating properly wrapped Atari environment...")
    env = make_atari_env("PongNoFrameskip-v4", render_mode=render_mode)
    
    # Print environment information
    print(f"\nObservation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Run diagnostic tests
    try:
        # Test environment with random policy
        random_rewards, action_dist, random_steps = run_random_policy(env, num_episodes=2)
        
        # Test simple deterministic policies
        up_rewards, up_steps = run_simple_policy(env, policy_type="up", num_episodes=2)
        down_rewards, down_steps = run_simple_policy(env, policy_type="down", num_episodes=2)
        
        # Analyze state values
        analyze_state_values(env, num_steps=500)
        
        print("\n" + "=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)
        print(f"Random Policy Avg Reward: {np.mean(random_rewards):.2f}")
        print(f"Always Up Policy Avg Reward: {np.mean(up_rewards):.2f}")
        print(f"Always Down Policy Avg Reward: {np.mean(down_rewards):.2f}")
        print(f"Action Distribution: {dict(action_dist)}")
        print("\nCheck the visualizations in data/pong/diagnostics/ for more insights.")
        print("=" * 80)
        
    finally:
        # Close environment
        env.close()

if __name__ == "__main__":
    main()
