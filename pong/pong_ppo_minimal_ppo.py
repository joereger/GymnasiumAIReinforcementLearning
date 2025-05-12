"""
PPO algorithm implementation for Pong with detailed diagnostics.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from pong_ppo_minimal_env import device

class DiagnosticPPO:
    """
    Proximal Policy Optimization algorithm with built-in diagnostics.
    """
    def __init__(
        self,
        actor_critic,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        normalize_advantage=True,
        diagnostic_mode=True
    ):
        self.actor_critic = actor_critic
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
        
        # Add learning rate scheduler (reduce LR by 10% every 12 updates, ~50K steps)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=12, gamma=0.9)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.normalize_advantage = normalize_advantage
        self.diagnostic_mode = diagnostic_mode
        self.diagnostic_dir = "data/pong/diagnostics"
        os.makedirs(f"{self.diagnostic_dir}/ppo", exist_ok=True)
        
        # Track metrics for diagnostics
        self.update_count = 0
        self.advantage_history = []
        self.return_history = []
        self.ratio_history = []
        self.policy_loss_history = []
        self.value_loss_history = []
        self.entropy_history = []
        self.grad_norm_history = []
        self.action_probs_history = []
    
    def compute_gae(self, rewards, values, dones, next_value, debug=False):
        """
        Compute Generalized Advantage Estimation with diagnostic output.
        Fixed to properly handle episode boundaries and maintain numerical stability.
        
        Args:
            rewards: List of rewards in a rollout
            values: List of value estimates in a rollout
            dones: List of episode termination flags
            next_value: Value estimate for the state after the rollout
            debug: Whether to print debug information
            
        Returns:
            advantages: GAE advantages
            returns: Discounted returns (value targets)
        """
        rollout_len = len(rewards)
        advantages = np.zeros(rollout_len, dtype=np.float32)
        returns = np.zeros(rollout_len, dtype=np.float32)
        
        # The last advantage is based on the next value
        last_gae_lam = 0
        
        # For diagnostics
        if debug:
            deltas = np.zeros(rollout_len, dtype=np.float32)
        
        # Critical: Loop backwards through the rollout for GAE calculation
        for t in reversed(range(rollout_len)):
            # Handle episode boundaries properly by zeroing out the advantage at end of episodes
            if dones[t]:
                next_value_t = 0  # Terminal state has zero value
                next_non_terminal = 0.0  # Terminal state is definitely terminal
            else:
                # If this is the last step of rollout but not end of episode, use next_value
                if t == rollout_len - 1:
                    next_value_t = next_value
                else:
                    next_value_t = values[t + 1]
                next_non_terminal = 1.0
                
            # Calculate TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            
            # Store delta for diagnostics
            if debug:
                deltas[t] = delta
            
            # Compute GAE recursively with proper discounting at episode boundaries
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
            # Compute returns (used as value targets)
            if dones[t]:
                # For terminal states, the return is just the reward
                returns[t] = rewards[t]
                # Reset GAE for new episode
                last_gae_lam = 0
            else:
                # For non-terminal states, compute return normally
                returns[t] = advantages[t] + values[t]
        
        # For diagnostics: log GAE components at the start and then every ~50k steps
        if debug:
            # Only at the start and approximately every 12 updates (50k steps)
            if self.update_count == 0 or self.update_count % 12 == 0:
                # Save diagnostic information
                plt.figure(figsize=(12, 8))
                plt.subplot(3, 1, 1)
                plt.plot(rewards, label='Rewards')
                plt.legend()
                plt.title('Rewards')
                
                plt.subplot(3, 1, 2)
                plt.plot(values, label='Values')
                plt.plot(returns, label='Returns')
                plt.legend()
                plt.title('Values and Returns')
                
                plt.subplot(3, 1, 3)
                plt.plot(deltas, label='TD Errors')
                plt.plot(advantages, label='Advantages')
                plt.legend()
                plt.title('TD Errors and Advantages')
                
                plt.tight_layout()
                plt.savefig(f"{self.diagnostic_dir}/ppo/gae_components_{self.update_count}.png")
                plt.close()
            
            # Log stats to file
            with open(f"{self.diagnostic_dir}/ppo/gae_stats.txt", "a") as f:
                f.write(f"Update {self.update_count}:\n")
                f.write(f"  Rewards: min={np.min(rewards):.4f}, max={np.max(rewards):.4f}, mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}\n")
                f.write(f"  Values: min={np.min(values):.4f}, max={np.max(values):.4f}, mean={np.mean(values):.4f}, std={np.std(values):.4f}\n")
                f.write(f"  Returns: min={np.min(returns):.4f}, max={np.max(returns):.4f}, mean={np.mean(returns):.4f}, std={np.std(returns):.4f}\n")
                f.write(f"  Deltas: min={np.min(deltas):.4f}, max={np.max(deltas):.4f}, mean={np.mean(deltas):.4f}, std={np.std(deltas):.4f}\n")
                f.write(f"  Advantages: min={np.min(advantages):.4f}, max={np.max(advantages):.4f}, mean={np.mean(advantages):.4f}, std={np.std(advantages):.4f}\n\n")
        
        return advantages, returns
    
    def save_action_distribution(self, action_probs):
        """Save visualization of action probability distribution."""
        # Only save at start and then every ~50k steps (12 updates)
        should_save = (self.update_count == 0 or self.update_count % 12 == 0)
        
        # Compute the average action distribution
        avg_probs = np.mean(action_probs, axis=0)
        
        # Only save the visualization if it's at the right frequency
        if should_save:
            plt.figure(figsize=(10, 6))
            plt.bar(['NOOP', 'UP', 'DOWN'], avg_probs)
            plt.ylim(0, 1)
            plt.title(f'Average Action Distribution (Update {self.update_count})')
            plt.ylabel('Probability')
            plt.savefig(f"{self.diagnostic_dir}/ppo/action_dist_{self.update_count}.png")
            plt.close()
        
        # Always calculate the entropy of the average distribution and log stats
        entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10))
        
        # Log stats to file
        with open(f"{self.diagnostic_dir}/ppo/action_dist_stats.txt", "a") as f:
            f.write(f"Update {self.update_count}: Probs=[{avg_probs[0]:.4f}, {avg_probs[1]:.4f}, {avg_probs[2]:.4f}], Entropy={entropy:.4f}\n")
    
    def update(self, rollout, n_epochs=4, batch_size=64):
        """
        Update policy and value function using PPO with detailed diagnostics.
        
        Args:
            rollout: Dictionary containing collected experience
            n_epochs: Number of optimization epochs per update
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
        
        # For diagnostic tracking
        all_ratios = []
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_action_probs = []
        
        # Track gradient metrics
        all_grad_norms = []
        
        # Store original advantage stats for diagnostics
        if self.diagnostic_mode:
            orig_adv_mean = advantages.mean().item()
            orig_adv_std = advantages.std().item()
            orig_adv_min = advantages.min().item()
            orig_adv_max = advantages.max().item()
            
            # Log advantage and return statistics
            self.advantage_history.append({
                'update': self.update_count,
                'mean': orig_adv_mean,
                'std': orig_adv_std,
                'min': orig_adv_min,
                'max': orig_adv_max
            })
            
            self.return_history.append({
                'update': self.update_count,
                'mean': returns.mean().item(),
                'std': returns.std().item(),
                'min': returns.min().item(),
                'max': returns.max().item()
            })
        
        # Normalize advantages with better numerical stability
        if self.normalize_advantage and len(advantages) > 1:
            # Use a larger epsilon for stability and check if std is too small
            adv_std = advantages.std()
            if adv_std < 1e-5:
                # If std is too small, don't normalize
                print(f"Warning: Advantage std too small ({adv_std:.8f}), skipping normalization")
            else:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
            
            # Safety check for NaN or inf values
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                print("Warning: NaN or Inf detected in normalized advantages, resetting to original")
                advantages = torch.FloatTensor(rollout['advantages']).to(device)
            else:
                if self.diagnostic_mode:
                    norm_adv_mean = advantages.mean().item()
                    norm_adv_std = advantages.std().item()
                    
                    # Log normalization effect
                    with open(f"{self.diagnostic_dir}/ppo/advantage_norm.txt", "a") as f:
                        f.write(f"Update {self.update_count}: " +
                               f"Before: mean={orig_adv_mean:.4f}, std={orig_adv_std:.4f}, " +
                               f"min={orig_adv_min:.4f}, max={orig_adv_max:.4f} | " +
                               f"After: mean={norm_adv_mean:.4f}, std={norm_adv_std:.4f}\n")
        
        # Multiple optimization epochs
        for epoch in range(n_epochs):
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
                
                # Get current log probs, values, and entropy
                mb_new_log_probs, mb_values, entropy, mb_action_probs = self.actor_critic.evaluate_actions(mb_states, mb_actions)
                
                # Store action probs for diagnostics
                if self.diagnostic_mode:
                    all_action_probs.append(mb_action_probs.detach().cpu().numpy())
                
                # Compute policy loss with improved numerical stability
                # Clip log probability differences to prevent extreme ratio values
                log_prob_diff = mb_new_log_probs - mb_old_log_probs
                log_prob_diff = torch.clamp(log_prob_diff, -20.0, 20.0)  # Prevent extreme values
                ratio = torch.exp(log_prob_diff)
                
                # Safety check for NaN or inf values in ratios
                if torch.isnan(ratio).any() or torch.isinf(ratio).any():
                    print("Warning: NaN or Inf detected in policy ratios, clamping values")
                    ratio = torch.clamp(ratio, 0.05, 20.0)  # Reasonable bounds for ratios
                
                # Store ratios for diagnostics
                if self.diagnostic_mode:
                    all_ratios.extend(ratio.detach().cpu().numpy())
                
                # PPO clipped objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss (MSE)
                value_loss = nn.functional.mse_loss(mb_values, mb_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Store losses for diagnostics
                if self.diagnostic_mode:
                    all_policy_losses.append(policy_loss.item())
                    all_value_losses.append(value_loss.item())
                    all_entropies.append(entropy.item())
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                
                # Compute gradient norm for diagnostics
                if self.diagnostic_mode:
                    grad_norm = 0.0
                    for param in self.actor_critic.parameters():
                        if param.grad is not None:
                            grad_norm += param.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    all_grad_norms.append(grad_norm)
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
        
        # Compute and log diagnostic metrics
        if self.diagnostic_mode:
            # Save action probability distribution
            if len(all_action_probs) > 0:
                all_action_probs = np.concatenate([probs for probs in all_action_probs if len(probs) > 0])
                self.save_action_distribution(all_action_probs)
            
            # Calculate and store average metrics
            avg_policy_loss = np.mean(all_policy_losses)
            avg_value_loss = np.mean(all_value_losses)
            avg_entropy = np.mean(all_entropies)
            avg_grad_norm = np.mean(all_grad_norms)
            
            self.policy_loss_history.append(avg_policy_loss)
            self.value_loss_history.append(avg_value_loss)
            self.entropy_history.append(avg_entropy)
            self.grad_norm_history.append(avg_grad_norm)
            
            # Log ratios
            if len(all_ratios) > 0:
                ratio_mean = np.mean(all_ratios)
                ratio_min = np.min(all_ratios)
                ratio_max = np.max(all_ratios)
                ratio_std = np.std(all_ratios)
                
                self.ratio_history.append({
                    'update': self.update_count,
                    'mean': ratio_mean,
                    'std': ratio_std,
                    'min': ratio_min,
                    'max': ratio_max
                })
                
                # Plot ratio distribution - only at start and then every ~50k steps
                if self.update_count == 0 or self.update_count % 12 == 0:  # Keep at 50k steps (12 updates)
                    plt.figure(figsize=(10, 6))
                    plt.hist(all_ratios, bins=50, alpha=0.7)
                    plt.axvline(x=1.0, color='r', linestyle='--')
                    plt.axvline(x=1.0 + self.clip_epsilon, color='g', linestyle='--')
                    plt.axvline(x=1.0 - self.clip_epsilon, color='g', linestyle='--')
                    plt.title(f'Policy Update Ratio Distribution (Update {self.update_count})')
                    plt.xlabel('Ratio (π/π_old)')
                    plt.ylabel('Count')
                    plt.savefig(f"{self.diagnostic_dir}/ppo/ratio_dist_{self.update_count}.png")
                    plt.close()
                
                # Always log ratio information to file - text logs don't take much space
                with open(f"{self.diagnostic_dir}/ppo/ratio_stats.txt", "a") as f:
                    f.write(f"Update {self.update_count}: " +
                           f"mean={ratio_mean:.4f}, std={ratio_std:.4f}, " +
                           f"min={ratio_min:.4f}, max={ratio_max:.4f}\n")
            
            # Save plots of loss histories - only at start and then every ~50k steps
            if self.update_count == 0 or self.update_count % 12 == 0:
                updates = list(range(self.update_count + 1))
                
                # Plot loss metrics
                plt.figure(figsize=(15, 12))
                
                plt.subplot(3, 2, 1)
                plt.plot(updates, self.policy_loss_history)
                plt.title('Policy Loss')
                plt.xlabel('Update')
                
                plt.subplot(3, 2, 2)
                plt.plot(updates, self.value_loss_history)
                plt.title('Value Loss')
                plt.xlabel('Update')
                
                plt.subplot(3, 2, 3)
                plt.plot(updates, self.entropy_history)
                plt.title('Entropy')
                plt.xlabel('Update')
                
                plt.subplot(3, 2, 4)
                plt.plot(updates, self.grad_norm_history)
                plt.title('Gradient Norm')
                plt.xlabel('Update')
                
                # Plot advantage means and stds
                plt.subplot(3, 2, 5)
                adv_means = [x['mean'] for x in self.advantage_history]
                adv_stds = [x['std'] for x in self.advantage_history]
                plt.plot(updates, adv_means, label='Mean')
                plt.fill_between(updates, 
                                 [m - s for m, s in zip(adv_means, adv_stds)],
                                 [m + s for m, s in zip(adv_means, adv_stds)],
                                 alpha=0.2)
                plt.title('Advantage Mean & Std')
                plt.xlabel('Update')
                plt.legend()
                
                # Plot ratio means and clip bounds
                plt.subplot(3, 2, 6)
                if len(self.ratio_history) > 0:
                    ratio_means = [x['mean'] for x in self.ratio_history]
                    ratio_stds = [x['std'] for x in self.ratio_history]
                    ratio_updates = list(range(len(ratio_means)))
                    plt.plot(ratio_updates, ratio_means, label='Mean Ratio')
                    plt.fill_between(ratio_updates,
                                     [m - s for m, s in zip(ratio_means, ratio_stds)],
                                     [m + s for m, s in zip(ratio_means, ratio_stds)],
                                     alpha=0.2)
                    plt.axhline(y=1.0, color='r', linestyle='--', label='Baseline')
                    plt.axhline(y=1.0 + self.clip_epsilon, color='g', linestyle='--', label='Clip Bounds')
                    plt.axhline(y=1.0 - self.clip_epsilon, color='g', linestyle='--')
                    plt.title('Policy Update Ratio')
                    plt.xlabel('Update')
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig(f"{self.diagnostic_dir}/ppo/training_metrics.png")
                plt.close()
        
        # Increment update counter
        self.update_count += 1
        
        # Step the learning rate scheduler
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        
        # Log the learning rate
        if self.diagnostic_mode:
            with open(f"{self.diagnostic_dir}/ppo/learning_rates.txt", "a") as f:
                f.write(f"Update {self.update_count}: Learning rate = {current_lr:.6f}\n")
        
        # Compute average metrics
        avg_policy_loss = np.mean(all_policy_losses)
        avg_value_loss = np.mean(all_value_losses)
        avg_entropy = np.mean(all_entropies)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'grad_norm': np.mean(all_grad_norms) if len(all_grad_norms) > 0 else 0.0
        }
