import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# Determine device for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# --- Actor-Critic Network Architecture ---
class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for PPO.
    
    Features:
    - Shared CNN feature extractor for both policy and value functions
    - Actor (policy) network outputs action probabilities
    - Critic (value) network estimates state values
    """
    def __init__(self, input_channels=4, action_space=6):
        super(ActorCritic, self).__init__()
        
        # Shared CNN feature extractor (same architecture as DQN)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of flattened features
        self.fc_input_dims = 7 * 7 * 64
        
        # Actor (Policy) Network
        self.actor = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )
        
        # Critic (Value) Network
        self.critic = nn.Sequential(
            nn.Linear(self.fc_input_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        # Extract features from input state
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Compute action probabilities (actor)
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Compute state value (critic)
        state_values = self.critic(features)
        
        return action_probs, state_values
    
    def evaluate(self, states, actions):
        """
        Evaluate the log probabilities of actions and the state values for given states.
        
        Args:
            states: Batch of states [batch_size, channels, height, width]
            actions: Batch of actions [batch_size]
            
        Returns:
            log_probs: Log probabilities of the actions taken
            state_values: Value function estimates
            entropy: Entropy of the policy distribution
        """
        action_probs, state_values = self(states)
        
        # Create a distribution from action probabilities
        dist = Categorical(action_probs)
        
        # Get log probabilities of actions
        log_probs = dist.log_prob(actions)
        
        # Calculate entropy of the policy (for exploration bonus)
        entropy = dist.entropy()
        
        return log_probs, state_values.squeeze(), entropy

# --- PPO Agent ---
class PPOAgent:
    """
    PPO agent for training and evaluation.
    
    Implements:
    - PPO clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Multiple optimization epochs on batched data
    - Entropy regularization
    """
    def __init__(self, 
                 state_shape, 
                 action_space,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_param=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 ppo_epochs=4,
                 batch_size=32):
        """
        Initialize PPO agent.
        
        Args:
            state_shape: Shape of the state observations (e.g., (4, 84, 84) for stacked frames)
            action_space: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_param: PPO clipping parameter (epsilon)
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of optimization epochs per update
            batch_size: Minibatch size for updates
        """
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Create Actor-Critic network
        self.ac_network = ActorCritic(input_channels=state_shape[0], action_space=action_space).to(device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.ac_network.parameters(), lr=lr)
        
        # Step counter
        self.steps = 0
        
        # For training stats
        self.loss_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": []
        }
    
    def act(self, state, deterministic=False):
        """
        Select an action given a state.
        
        Args:
            state: Current state observation [channels, height, width]
            deterministic: If True, select the action with highest probability (for evaluation)
        
        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            value: Value estimate for the state
        """
        self.steps += 1
        
        # Ensure state has correct format (channels first for PyTorch)
        if len(state.shape) == 3 and state.shape[0] != self.state_shape[0]:
            # Convert from (H, W, C) to (C, H, W) if needed
            state = np.transpose(state, (2, 0, 1))
            
        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Get action probabilities and state value
        with torch.no_grad():
            action_probs, state_value = self.ac_network(state_tensor)
        
        # Select action
        if deterministic:
            # For evaluation, select action with highest probability
            action = torch.argmax(action_probs, dim=1).item()
            # Still need log prob for stats
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(torch.tensor([action], device=device)).item()
        else:
            # Sample action from distribution for exploration
            dist = Categorical(action_probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor([action], device=device)).item()
        
        return action, log_prob, state_value.item()
    
    def update(self, rollout_data):
        """
        Update policy and value network using PPO.
        
        Args:
            rollout_data: Dictionary of rollout data from get_batch()
            
        Returns:
            Dictionary of loss statistics
        """
        # Clear stats for this update
        for key in self.loss_stats:
            self.loss_stats[key] = []
        
        # Multiple optimization epochs
        for _ in range(self.ppo_epochs):
            # If mini-batch size is specified, create mini-batches
            if self.batch_size is not None and self.batch_size < len(rollout_data["states"]):
                # Generate random mini-batch indices
                indices = torch.randperm(len(rollout_data["states"]))
                
                # Process mini-batches
                for start_idx in range(0, len(indices), self.batch_size):
                    # Extract mini-batch indices
                    batch_indices = indices[start_idx:start_idx + self.batch_size]
                    
                    # Get mini-batch data
                    mini_batch = {
                        "states": rollout_data["states"][batch_indices],
                        "actions": rollout_data["actions"][batch_indices],
                        "log_probs_old": rollout_data["log_probs_old"][batch_indices],
                        "returns": rollout_data["returns"][batch_indices],
                        "advantages": rollout_data["advantages"][batch_indices],
                        "values": rollout_data["values"][batch_indices]
                    }
                    
                    # Perform update on mini-batch
                    self._update_mini_batch(mini_batch)
            else:
                # Use entire batch for update
                self._update_mini_batch(rollout_data)
        
        # Return average loss statistics for this update
        return {k: sum(v) / max(len(v), 1) for k, v in self.loss_stats.items()}
    
    def _update_mini_batch(self, mini_batch):
        """
        Perform one optimization step on a mini-batch of data.
        
        Args:
            mini_batch: Dictionary of mini-batch data
        """
        # Evaluate actions with current policy
        log_probs_new, values_new, entropy = self.ac_network.evaluate(
            mini_batch["states"], mini_batch["actions"]
        )
        
        # Compute policy (actor) loss
        
        # Calculate probability ratio
        # ratio = exp(log_prob_new - log_prob_old)
        ratio = torch.exp(log_probs_new - mini_batch["log_probs_old"])
        
        # Compute surrogate objectives
        surr1 = ratio * mini_batch["advantages"]
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * mini_batch["advantages"]
        
        # Clipped surrogate objective (negative for gradient ascent)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value (critic) loss
        # Use clipped value loss (from PPO implementation details)
        values_pred_clipped = mini_batch["values"] + torch.clamp(
            values_new - mini_batch["values"],
            -self.clip_param,
            self.clip_param
        )
        value_loss1 = F.mse_loss(values_new, mini_batch["returns"])
        value_loss2 = F.mse_loss(values_pred_clipped, mini_batch["returns"])
        value_loss = 0.5 * torch.max(value_loss1, value_loss2)
        
        # Compute entropy bonus (for exploration)
        entropy_loss = -entropy.mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Record loss statistics
        self.loss_stats["policy_loss"].append(policy_loss.item())
        self.loss_stats["value_loss"].append(value_loss.item())
        self.loss_stats["entropy"].append(-entropy_loss.item())  # Negative sign to make it positive
        self.loss_stats["total_loss"].append(total_loss.item())
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.ac_network.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
    
    def save(self, path):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.ac_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=device)
        self.ac_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        print(f"Model loaded from {path}, steps: {self.steps}")
