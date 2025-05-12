"""
PPO model architecture for Pong.
Implements the actor-critic network and PPO algorithm components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from pong_env_wrappers import device

class PPOActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO.
    
    The architecture follows a CNN backbone (shared between actor and critic)
    followed by separate policy (actor) and value (critic) heads.
    """
    def __init__(self, input_channels=4, action_dim=3):
        super(PPOActorCritic, self).__init__()
        
        # CNN backbone - processes 84x84 images with 4 stacked frames
        self.cnn_base = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output size (7x7x64 = 3136 for 84x84 input)
        cnn_output_size = self._get_conv_output_size(input_channels, 84, 84)
        
        # Actor (policy) head - outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
        # Critic (value) head - outputs state value estimation
        self.critic = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_output_size(self, input_channels, height, width):
        """Calculate the output size of the CNN backbone."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, height, width)
            output = self.cnn_base(dummy_input)
            return output.shape[1]
    
    def forward(self, x):
        """
        Forward pass through the network.
        Returns action logits and state value.
        """
        # Ensure the input is on the right device
        x = x.to(device)
        
        # Process through CNN backbone
        features = self.cnn_base(x)
        
        # Get action logits and state value
        action_logits = self.actor(features)
        state_value = self.critic(features)
        
        return action_logits, state_value
    
    def get_action(self, state, deterministic=False):
        """
        Sample an action from the policy distribution.
        
        Args:
            state: Environment state tensor (should be preprocessed)
            deterministic: If True, return the most probable action instead of sampling
            
        Returns:
            action: Selected action
            log_prob: Log probability of the selected action
            value: Estimated state value
        """
        # Add batch dimension if not present
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        # Forward pass
        action_logits, state_value = self(state)
        
        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Sample action or take most probable action
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        
        # Get log probability of the action
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), state_value.item()
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate actions for given states.
        Used during PPO update to compute loss.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            action_log_probs: Log probabilities of the actions
            state_values: Estimated state values
            entropy: Entropy of the action distribution (for exploration)
        """
        action_logits, state_values = self(states)
        
        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Get log probabilities, entropy, and state values
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        return action_log_probs, state_values.squeeze(-1), entropy


class PPO:
    """
    Proximal Policy Optimization algorithm.
    
    Implements the PPO-Clip variant with:
    - Clipped surrogate objective
    - Value function loss
    - Entropy bonus for exploration
    """
    def __init__(
        self,
        actor_critic,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        use_gae=True,
        normalize_advantage=True
    ):
        """
        Initialize PPO algorithm.
        
        Args:
            actor_critic: Actor-critic neural network
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for rewards
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            use_gae: Whether to use Generalized Advantage Estimation
            normalize_advantage: Whether to normalize advantages
        """
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.use_gae = use_gae
        self.normalize_advantage = normalize_advantage
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards in a rollout
            values: List of value estimates in a rollout
            dones: List of episode termination flags
            next_value: Value estimate for the state after the rollout
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns (value targets)
        """
        rollout_len = len(rewards)
        advantages = np.zeros(rollout_len, dtype=np.float32)
        returns = np.zeros(rollout_len, dtype=np.float32)
        
        # The last advantage is based on the next value
        last_gae_lam = 0
        
        # Loop backwards through the rollout
        for t in reversed(range(rollout_len)):
            # If this is the last step, use next_value, otherwise use values[t+1]
            next_non_terminal = 1.0 - dones[t]
            
            if t == rollout_len - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            # Calculate TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            
            # Compute GAE recursively
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            # Compute returns (used as value targets)
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, rollout, n_epochs=4, batch_size=64):
        """
        Update policy and value function using PPO.
        
        Args:
            rollout: Dictionary containing collected experience
                - 'states': Batch of states
                - 'actions': Batch of actions
                - 'old_log_probs': Log probs of actions at time of collection
                - 'advantages': Advantage estimates
                - 'returns': Discounted returns
            n_epochs: Number of optimization epochs
            batch_size: Mini-batch size for optimization
            
        Returns:
            loss_metrics: Dictionary of loss metrics
        """
        # Extract rollout data
        states = torch.FloatTensor(rollout['states']).to(device)
        actions = torch.LongTensor(rollout['actions']).to(device)
        old_log_probs = torch.FloatTensor(rollout['old_log_probs']).to(device)
        advantages = torch.FloatTensor(rollout['advantages']).to(device)
        returns = torch.FloatTensor(rollout['returns']).to(device)
        
        # Normalize advantages (reduces variance)
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Track metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        # Multiple optimization epochs
        for _ in range(n_epochs):
            # Generate random permutation for mini-batches
            indices = torch.randperm(len(states))
            
            # Mini-batch update
            for start_idx in range(0, len(states), batch_size):
                # Get mini-batch indices
                idx = indices[start_idx:start_idx + batch_size]
                
                # Mini-batch data
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_advantages = advantages[idx]
                mb_returns = returns[idx]
                
                # Get current log probs and values
                mb_new_log_probs, mb_values, entropy = self.actor_critic.evaluate_actions(mb_states, mb_actions)
                
                # Compute policy loss (clipped surrogate objective)
                ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(mb_values, mb_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Compute average metrics
        num_updates = n_epochs * (len(states) // batch_size + 1)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy
        }
