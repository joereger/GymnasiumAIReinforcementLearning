"""
Environment wrappers and setup for Pong PPO implementation.
"""

import os
import gymnasium as gym
import numpy as np
import torch
import cv2
from collections import deque
import matplotlib.pyplot as plt

# Get appropriate device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

class DiagnosticsWrapper(gym.Wrapper):
    """
    Wrapper to capture and visualize frames at different processing stages.
    """
    def __init__(self, env, visualization_freq=500):
        super(DiagnosticsWrapper, self).__init__(env)
        self.visualization_freq = visualization_freq
        self.step_counter = 0
        self.diagnostic_dir = "data/pong/diagnostics"
        os.makedirs(self.diagnostic_dir, exist_ok=True)
        
    def step(self, action):
        """Take a step in the environment with diagnostics."""
        self.step_counter += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Only visualize at start and then every 50000 steps
        if self.step_counter == 0 or self.step_counter % 50000 == 0:
            self._visualize_observation(obs, f"step_{self.step_counter}")
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visualize_observation(obs, "reset")
        return obs, info
    
    def _visualize_observation(self, obs, label):
        """Save visualization of observation."""
        try:
            # For frame stack (4, 84, 84) - channels first
            if len(obs.shape) == 3 and obs.shape[0] == 4:
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                for i in range(4):
                    axes[i].imshow(obs[i], cmap='gray')
                    axes[i].set_title(f"Frame {i+1}")
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{self.diagnostic_dir}/obs_{label}.png")
                plt.close()
                
                # Also save a single frame
                plt.figure(figsize=(8, 8))
                plt.imshow(obs[3], cmap='gray')  # Most recent frame
                plt.title("Most Recent Frame")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f"{self.diagnostic_dir}/frame_{label}.png")
                plt.close()
                
                # Log observation stats
                with open(f"{self.diagnostic_dir}/obs_stats.txt", "a") as f:
                    f.write(f"{label}: shape={obs.shape}, dtype={obs.dtype}, " +
                            f"min={obs.min():.4f}, max={obs.max():.4f}, " +
                            f"mean={obs.mean():.4f}, std={obs.std():.4f}\n")
        except Exception as e:
            print(f"Error in observation visualization: {e}")


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
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
        obs, _, terminated, truncated, info = self.env.step(2)  # RIGHT
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
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
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted."""
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
    Return only every `skip`-th frame and max over the last 2 frames.
    """
    def __init__(self, env, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
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
    Clip rewards to {-1, 0, 1} for stability.
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
        self.observation_space = new_space

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


class ChannelsFirstImageShape(gym.ObservationWrapper):
    """
    Transpose observation from HWC to CHW format, specifically for PyTorch.
    """
    def __init__(self, env):
        super(ChannelsFirstImageShape, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[2], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=new_shape, 
            dtype=np.uint8
        )

    def observation(self, observation):
        return np.transpose(observation, (2, 0, 1))
        

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
        return np.array(observation).astype(np.float32) / 255.0


class ReducedActionSpace(gym.ActionWrapper):
    """
    Reduce the action space for Pong from 6 to 3 actions:
    NOOP(0), RIGHT(2), LEFT(3) -> [0, 1, 2]
    These correspond to STAY, PADDLE UP, PADDLE DOWN in Pong.
    """
    def __init__(self, env):
        super(ReducedActionSpace, self).__init__(env)
        self.valid_actions = [0, 2, 3]  # NOOP, RIGHT, LEFT
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))
        print(f"Reduced action space from {len(env.unwrapped.get_action_meanings())} to {len(self.valid_actions)} actions")
        print(f"Using actions: {[env.unwrapped.get_action_meanings()[a] for a in self.valid_actions]}")

    def action(self, action):
        return self.valid_actions[action]


def make_pong_env(env_id="PongNoFrameskip-v4", render_mode=None, reduced_actions=True, seed=None):
    """
    Create a preprocessed Pong environment with essential wrappers and diagnostics.
    
    Args:
        env_id: Environment ID (should be "PongNoFrameskip-v4")
        render_mode: Rendering mode (None or "human")
        reduced_actions: Whether to reduce action space to 3 actions
        seed: Random seed
        
    Returns:
        Wrapped gymnasium environment
    """
    try:
        import ale_py
        gym.register_envs(ale_py)
    except (ImportError, AttributeError) as e:
        print(f"Warning: {e}. Continuing without explicit ALE registration.")
    
    # Create base environment
    env = gym.make(
        env_id, 
        render_mode=render_mode, 
        repeat_action_probability=0.0, 
        full_action_space=False,
        disable_env_checker=True
    )
    
    # Disable sound if possible
    if hasattr(env.unwrapped, 'ale') and hasattr(env.unwrapped.ale, 'setInt'):
        env.unwrapped.ale.setInt('sound', 0)  # Turn off sound
    
    # Print original observation and action spaces
    print(f"Original observation space: {env.observation_space}")
    print(f"Original action space: {env.action_space}")
    print(f"Action meanings: {env.unwrapped.get_action_meanings()}")
    
    # Apply essential wrappers
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = ChannelsFirstImageShape(env)  # CHW format for PyTorch
    env = ScaledFloatFrame(env)
    
    if reduced_actions:
        env = ReducedActionSpace(env)
    
    # Add diagnostics wrapper
    env = DiagnosticsWrapper(env, visualization_freq=1000)
    
    # Print transformed observation and action spaces
    print(f"Transformed observation space: {env.observation_space}, dtype: {env.observation_space.dtype}")
    print(f"Transformed action space: {env.action_space}")
    
    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)
    
    return env
