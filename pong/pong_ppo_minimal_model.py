"""
Neural network model architecture for Pong PPO implementation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

# Import device from env module
from pong_ppo_minimal_env import device

class DiagnosticActorCritic(nn.Module):
    """
    Combined actor-critic network with diagnostic capabilities.
    Includes visualization of activations and gradient tracking.
    """
    def __init__(self, input_channels=4, action_dim=3):
        super(DiagnosticActorCritic, self).__init__()
        
        # CNN backbone with improved architecture to prevent collapse
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.batchnorm1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        
        # Use orthogonal initialization with adjusted gain values
        nn.init.orthogonal_(self.conv1.weight, gain=0.8)  # Slightly lower gain
        nn.init.orthogonal_(self.conv2.weight, gain=0.8)
        nn.init.orthogonal_(self.conv3.weight, gain=0.8)
        
        # Calculate CNN output size
        self.feature_size = self._get_conv_output_size(input_channels)
        
        # Actor (policy) head with leaky ReLU instead of ReLU
        self.actor = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, action_dim)
        )
        
        # Critic (value) head with leaky ReLU
        self.critic = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 1)
        )
        
        # Initialize linear layers properly
        nn.init.orthogonal_(self.actor[0].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.actor[0].bias, 0.0)
        nn.init.orthogonal_(self.actor[2].weight, gain=0.01)  # Small weight for final layer is critical
        nn.init.constant_(self.actor[2].bias, 0.0)
        
        nn.init.orthogonal_(self.critic[0].weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.critic[0].bias, 0.0)
        nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
        nn.init.constant_(self.critic[2].bias, 0.0)
        
        # For diagnostics
        self.activation = {}
        self.register_hooks()
        self.diagnostic_dir = "data/pong/diagnostics"
        os.makedirs(f"{self.diagnostic_dir}/activations", exist_ok=True)
        self.diagnostic_counter = 0
        
        # Generate a dummy input to capture initial activations at step 0
        # Move model to device first, then create dummy input
        self.to(device)
        dummy_input = torch.zeros(1, input_channels, 84, 84, device=device)
        self(dummy_input)  # Forward pass to register activations
        self._visualize_activations(0)  # Explicitly visualize at step 0
        
    def register_hooks(self):
        """Register hooks to capture activations."""
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output.detach()
            return hook
        
        self.conv1.register_forward_hook(get_activation('conv1'))
        self.conv2.register_forward_hook(get_activation('conv2'))
        self.conv3.register_forward_hook(get_activation('conv3'))
        
    def _get_conv_output_size(self, input_channels):
        """Calculate the output size of the CNN backbone."""
        # Create a dummy batch with the same dimensions as the actual input - use CPU for this calculation
        dummy_input = torch.zeros(1, input_channels, 84, 84)
        # Make sure model stays on CPU during this calculation
        with torch.no_grad():
            # Pass through CNN layers to get the output shape - using the same architecture as forward()
            x = self.batchnorm1(F.leaky_relu(self.conv1(dummy_input), negative_slope=0.01))
            x = self.batchnorm2(F.leaky_relu(self.conv2(x), negative_slope=0.01))
            x = self.batchnorm3(F.leaky_relu(self.conv3(x), negative_slope=0.01))
            # Flatten and get the size
            x = x.flatten(1)
        print(f"CNN output shape: {x.shape} - flattened size: {x.shape[1]}")
        return x.shape[1]  # Return the flattened feature size
    
    def _visualize_activations(self, step):
        """
        Visualize CNN activations to understand feature extraction.
        Only saves after first step and then every 50,000 steps to save disk space.
        """
        # Only visualize at specific intervals
        if step != 0 and step % 50000 != 0:
            return
            
        for name, activation in self.activation.items():
            # Take first example in batch
            act = activation[0].cpu().numpy()
            
            # For convolutional layers
            if len(act.shape) == 3:
                # Only visualize a subset of filters to keep it manageable
                num_filters = min(16, act.shape[0])
                fig, axes = plt.subplots(4, 4, figsize=(10, 10))
                axes = axes.flatten()
                
                for i in range(num_filters):
                    im = axes[i].imshow(act[i], cmap='viridis')
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(f"{self.diagnostic_dir}/activations/{name}_step_{step}.png")
                plt.close()
                
                # Also save activation statistics
                with open(f"{self.diagnostic_dir}/activation_stats.txt", "a") as f:
                    f.write(f"Step {step}, {name}: shape={act.shape}, " +
                            f"min={act.min():.4f}, max={act.max():.4f}, " +
                            f"mean={act.mean():.4f}, std={act.std():.4f}\n")
    
    def forward(self, x):
        """Forward pass through the network with batch normalization and leaky ReLU."""
        # Ensure the input is on the right device
        x = x.to(device)
        
        # CNN feature extraction with batch normalization and leaky ReLU
        x = self.batchnorm1(F.leaky_relu(self.conv1(x), negative_slope=0.01))
        x = self.batchnorm2(F.leaky_relu(self.conv2(x), negative_slope=0.01))
        x = self.batchnorm3(F.leaky_relu(self.conv3(x), negative_slope=0.01))
        x = x.reshape(x.size(0), -1)  # Flatten
        
        # Get action logits and state value
        action_logits = self.actor(x)
        state_value = self.critic(x)
        
        # Periodically visualize activations (not every step to avoid slowdown)
        self.diagnostic_counter += 1
        if self.diagnostic_counter % 1000 == 0:
            self._visualize_activations(self.diagnostic_counter)
        
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
            entropy: Entropy of the action distribution
            action_probs: Full action probability distribution
        """
        # Add batch dimension if not present
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        # Forward pass
        action_logits, state_value = self(state)
        
        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        
        # Calculate entropy
        entropy = dist.entropy().item()
        
        # Sample action or take most probable action
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            action = dist.sample()
        
        # Get log probability of the action
        log_prob = dist.log_prob(action)
        
        return (
            action.item(), 
            log_prob.item(), 
            state_value.item(), 
            entropy,
            action_probs.squeeze().detach().cpu().numpy()
        )
    
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
            action_probs: Full action probability distributions
        """
        action_logits, state_values = self(states)
        
        # Clip action logits to prevent extreme values
        action_logits = torch.clamp(action_logits, -20.0, 20.0)
        
        # Create action distribution with improved numerical stability
        # Add a small epsilon to prevent zeros
        action_probs = F.softmax(action_logits, dim=-1)
        action_probs = action_probs.clamp(min=1e-8, max=1.0 - 1e-8)
        
        # Create categorical distribution
        dist = Categorical(action_probs)
        
        # Get log probabilities, entropy, and state values
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Check for NaNs in tensor values for debugging
        if torch.isnan(action_log_probs).any():
            print("NaN detected in action_log_probs")
            # Handle NaN values in log probs by replacing with small negative value
            action_log_probs = torch.nan_to_num(action_log_probs, nan=-10.0)
            
        if torch.isnan(entropy).any():
            print("NaN detected in entropy")
            # Handle NaN in entropy
            entropy = torch.tensor(0.01, device=device)
        
        return action_log_probs, state_values.squeeze(-1), entropy, action_probs
