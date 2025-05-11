import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# Determine device for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# --- DQN Model (Same architecture as before) ---
class PongDQN(nn.Module):
    """
    Deep Q-Network for Pong, with the same architecture as the original DQN implementation.
    """
    def __init__(self, input_channels=4, action_space=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(inplace=True),
        )
        self.fc_input_dims = 7 * 7 * 64
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512), nn.ReLU(inplace=True),
            nn.Linear(512, action_space),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# --- Double DQN Agent ---
class DoubleDQNAgent:
    """
    Agent that uses Double DQN to learn and select actions.
    
    Key differences from DQN:
    - Uses online network to select actions for target Q-value computation
    - Uses target network to evaluate those actions
    - Reduces overestimation bias in Q-values
    - Modified hyperparameters: lower learning rate, more frequent target updates
    """
    def __init__(self, state_shape, action_space, 
                 lr=1e-5,  # Reduced learning rate for Experiment 2
                 gamma=0.99, 
                 target_update_freq=1000,  # More frequent updates for Experiment 2
                 epsilon_start=1.0, 
                 epsilon_end=0.01, 
                 epsilon_decay_frames=2e6,  # Slower decay for Experiment 2
                 batch_size=32):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Create policy and target networks
        self.policy_net = PongDQN(input_channels=state_shape[0], action_space=action_space).to(device)
        self.target_net = PongDQN(input_channels=state_shape[0], action_space=action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Setup optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_frames = epsilon_decay_frames
        self.current_frames = 0

    def get_epsilon(self):
        """Calculate the current epsilon value for epsilon-greedy exploration."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.current_frames / self.epsilon_decay_frames)
        return epsilon

    def act(self, state, explore=True):
        """Select an action using the current policy with epsilon-greedy exploration."""
        if explore:
            self.current_frames += 1
        
        epsilon = self.get_epsilon() if explore else 0.0
        
        if explore and random.random() < epsilon:
            return random.randrange(self.action_space)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def learn(self, replay_buffer):
        """
        Update the policy network using Double DQN algorithm.
        
        Double DQN uses the online network to select actions and the target
        network to evaluate them, reducing overestimation bias.
        """
        if len(replay_buffer) < self.batch_size:
            return None

        # Sample a batch of transitions
        transitions = replay_buffer.sample(self.batch_size)
        batch = list(zip(*transitions)) 

        # Convert to PyTorch tensors
        state_batch = torch.FloatTensor(np.array(batch[0])).to(device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(device)
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1).to(device)

        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute next Q values using Double DQN approach
        with torch.no_grad():
            # Use online network to SELECT actions
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            
            # Use target network to EVALUATE those actions
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions)
            
            # Compute target Q values
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_state_values
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network if needed
        if self.current_frames % self.target_update_freq == 0 and self.current_frames > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

    def save(self, path):
        """Save the policy network weights to a file."""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """Load policy network weights and copy to target network."""
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
