import numpy as np
import cv2
import gymnasium as gym
import torch
import time
from collections import deque

# --- Atari Wrappers ---
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

class StackFrame(gym.Wrapper):
    """
    Stack the last k frames together and convert to pytorch format (channels first)
    """
    def __init__(self, env, k=4):
        super(StackFrame, self).__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        # Update observation space to have proper channels-first shape for PyTorch
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1.0,
            shape=(k, shp[0], shp[1]),  # (channels, height, width)
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(self._process_frame(obs))
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._process_frame(obs))
        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _process_frame(self, frame):
        # Extract the single channel if grayscale and normalize
        if frame.shape[-1] == 1:  # Check if it has a single channel dimension at the end
            frame = frame.squeeze(-1)  # Remove the channel dimension
        return frame.astype(np.float32) / 255.0

    def _get_stacked_obs(self):
        # Stack frames along first dimension (channels first for PyTorch)
        return np.stack(list(self.frames), axis=0)

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
    env = StackFrame(env, 4)

    return env

# --- Rollout Buffer ---
class RolloutBuffer:
    """Store transitions collected during PPO rollouts."""
    
    def __init__(self, buffer_size, state_shape, action_dim, device):
        """Initialize a rollout buffer for PPO.
        
        Args:
            buffer_size: Maximum number of timesteps to store in the buffer
            state_shape: Shape of states (e.g., (4, 84, 84) for stacked frames)
            action_dim: Number of possible actions
            device: PyTorch device
        """
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device
        
        # Initialize storage for rollout data
        self.states = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        
        # For GAE and returns calculation
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        # Pointer to current position in buffer
        self.idx = 0
        self.full = False
    
    def store(self, state, action, reward, done, log_prob, value):
        """Store a transition in the buffer."""
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.log_probs[self.idx] = log_prob
        self.values[self.idx] = value
        
        # Update pointer
        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True
    
    def compute_advantages_and_returns(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation (GAE) and returns.
        
        Args:
            last_value: Value estimate for the state after the last state in the buffer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter for advantage weighting
        """
        # For PTB length - 1 to 0
        # Length of the buffer is self.buffer_size if full, else self.idx
        buffer_len = self.buffer_size if self.full else self.idx
        
        if buffer_len == 0:
            return
        
        # Handle potential partial buffer - work with the valid portion only
        # In the case we have a non-full buffer, actual buffer len is self.idx (last valid idx + 1)
        last_gae = 0
        for step in reversed(range(buffer_len)):
            # If we're at the last step, use last_value for the next value
            # Otherwise, use the value from the next step
            if step == buffer_len - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            # TD error: reward + gamma * next_value * (1 - done) - current_value
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            
            # GAE formula: sum_t (gamma * lambda)^t * delta_{t+k}
            self.advantages[step] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            
            # Returns = advantages + values
            self.returns[step] = self.advantages[step] + self.values[step]
    
    def get_batch(self, batch_size=None):
        """Get a batch of data from the buffer.
        
        If batch_size is None, returns the entire buffer.
        Otherwise, returns a random sample of size batch_size.
        
        Returns:
            Dict of torch tensors containing the batch data
        """
        # Length of the buffer is self.buffer_size if full, else self.idx
        buffer_len = self.buffer_size if self.full else self.idx
        
        if buffer_len == 0:
            return None
        
        if batch_size is None:
            # Return entire buffer
            indices = np.arange(buffer_len)
        else:
            # Sample random indices
            indices = np.random.choice(buffer_len, min(buffer_len, batch_size), replace=False)
        
        # Extract data
        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        log_probs_old = torch.FloatTensor(self.log_probs[indices]).to(self.device)
        values = torch.FloatTensor(self.values[indices]).to(self.device)
        returns = torch.FloatTensor(self.returns[indices]).to(self.device)
        advantages = torch.FloatTensor(self.advantages[indices]).to(self.device)
        
        # Normalize advantages (helps with training stability)
        if len(advantages) > 1:
            # Only normalize if we have more than one sample
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Return as dict
        return {
            "states": states,
            "actions": actions,
            "log_probs_old": log_probs_old,
            "values": values,
            "returns": returns,
            "advantages": advantages
        }
    
    def clear(self):
        """Clear the buffer by resetting the pointer."""
        self.idx = 0
        self.full = False

# --- Helper to format time ---
def format_time(seconds):
    """Format seconds into a human-readable HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
