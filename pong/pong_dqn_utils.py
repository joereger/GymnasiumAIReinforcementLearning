import numpy as np
import cv2
import gymnasium as gym
from collections import deque
import random
import time

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

class ReducedActionSpace(gym.ActionWrapper):
    """
    Reduces the action space to only 3 relevant actions for Pong:
    NOOP (0), RIGHT/UP (2), LEFT/DOWN (3)
    
    This simplifies learning since the agent only needs to control
    vertical paddle movement.
    """
    def __init__(self, env):
        super(ReducedActionSpace, self).__init__(env)
        # Original actions: NOOP(0), FIRE(1), RIGHT(2), LEFT(3), RIGHTFIRE(4), LEFTFIRE(5)
        # For Pong:
        # - RIGHT (2) moves paddle UP
        # - LEFT (3) moves paddle DOWN
        # - NOOP (0) keeps paddle in place
        self.valid_actions = [0, 2, 3]  # [STAY, UP, DOWN]
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))
        
        # Store action meanings for debugging
        self.action_meanings = env.unwrapped.get_action_meanings()
        print(f"Reduced action space from {len(self.action_meanings)} to {len(self.valid_actions)} actions")
        print(f"Using actions: {[self.action_meanings[a] for a in self.valid_actions]}")
    
    def action(self, action):
        """Map the reduced action index to the original action."""
        return self.valid_actions[action]

def make_atari_env(env_id, render_mode=None, max_episode_steps=None, reduced_actions=True):
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
    
    # Optionally reduce action space to simplify learning
    if reduced_actions:
        env = ReducedActionSpace(env)

    return env

# --- Replay Buffer ---
class ReplayBuffer:
    """Store and sample transitions for DQN training."""
    def __init__(self, size=int(1e5)):
        self.buffer = deque(maxlen=size)

    def store(self, transition):
        """Store a transition (state, action, reward, next_state, done)."""
        self.buffer.append(transition)

    def sample(self, batch_size=32):
        """Sample a batch of transitions randomly."""
        if len(self.buffer) < batch_size:
            return []
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

# --- Helper to format time ---
def format_time(seconds):
    """Format seconds into a human-readable HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
