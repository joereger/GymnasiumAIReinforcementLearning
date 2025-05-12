"""
Environment wrappers for Pong based on the Atari literature standards.
These wrappers preprocess observations and adapt the environment for RL algorithms.
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import torch
import ale_py # Import ale_py

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

class PongEpisodicEnv(EpisodicLifeEnv):
    """
    Extension of EpisodicLifeEnv specifically for Pong.
    In Pong, game endings (21 points) don't register as life changes, so we need
    to detect large negative rewards to identify game endings.
    """
    def __init__(self, env):
        super(PongEpisodicEnv, self).__init__(env)
        self.score_accumulator = 0
        
    def step(self, action):
        # Use parent class step first
        obs, reward, terminated, truncated, info = super().step(action)
        
        # For Pong specifically, track large negative rewards
        # When raw reward is <= -21, it indicates a completed game
        # (opponent reached 21 points, which ends a game)
        self.score_accumulator += reward
        if self.score_accumulator <= -21:
            # Reset accumulator and trigger episode end
            self.score_accumulator = 0
            terminated = True
            
        return obs, reward, terminated, truncated, info
        
    def reset(self, **kwargs):
        # Reset score accumulator
        self.score_accumulator = 0
        return super().reset(**kwargs)

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
    Reduce the action space for Pong from 6 to 3 actions.
    The 3 actions are: NOOP(0), RIGHT(2), LEFT(3)
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
    Create a preprocessed Pong environment with all necessary wrappers.
    This version directly uses the specified env_id, defaulting to "PongNoFrameskip-v4".
    
    Args:
        env_id: Environment ID (should be "PongNoFrameskip-v4")
        render_mode: Rendering mode (None or "human")
        reduced_actions: Whether to reduce action space to 3 actions
        seed: Random seed
        
    Returns:
        Wrapped gymnasium environment
    """
    # Ensure ALE environments are registered
    gym.register_envs(ale_py)
    print(f"Attempting to create environment: {env_id}")
    
    # Create environment with sound disabled
    if render_mode == "human":
        print("Creating environment with sound disabled")
    
    env = gym.make(
        env_id, 
        render_mode=render_mode, 
        repeat_action_probability=0.0, 
        full_action_space=False,
        disable_env_checker=True
    )
    
    # Attempt to disable sound through ALE interface
    if hasattr(env.unwrapped, 'ale'):
        if hasattr(env.unwrapped.ale, 'setInt'):
            env.unwrapped.ale.setInt('sound', 0)  # Turn off sound
            # The display_screen setting is not available in this ALE version
    
    # Seeding will be handled by env.reset(seed=seed) in the training/evaluation scripts
        
    # Apply wrappers in the standard order
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    # Use the Pong-specific episodic environment wrapper for Pong
    if "Pong" in env_id:
        env = PongEpisodicEnv(env)
        print("Using PongEpisodicEnv wrapper for proper game boundary detection")
    else:
        env = EpisodicLifeEnv(env)
    
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = ChannelsFirstImageShape(env)  # PyTorch uses CHW format
    env = ScaledFloatFrame(env)
    
    if reduced_actions:
        env = ReducedActionSpace(env)
    
    return env

# Get appropriate device for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
