import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """
    Actor-Critic neural network architecture for reinforcement learning.
    Implements both the actor (policy) and critic (value function) networks.
    """
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim=128, 
        actor_lr=3e-4, 
        critic_lr=1e-3, 
        gamma=0.99, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        exploration_noise=0.5,
        min_exploration=0.05,
        exploration_decay=0.995
    ):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.device = device
        
        # Exploration parameters
        self.exploration_noise = exploration_noise
        self.min_exploration = min_exploration
        self.exploration_decay = exploration_decay
        self.training_steps = 0
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log_std for continuous actions
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + 
            list(self.mean_layer.parameters()) + 
            list(self.log_std_layer.parameters()), 
            lr=actor_lr
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Move model to device
        self.to(device)
        
        # For storing trajectory data
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Action statistics for debugging
        self.action_counts = {0: 0, 1: 0, 2: 0}  # buy, sell, hold
        
    def process_state(self, state):
        """
        Process state into the correct format for the network.
        """
        if isinstance(state, dict):
            # Extract relevant features from state dictionary
            price_history = np.array(state.get('price_history', []))
            portfolio_features = np.array([
                state.get('cash', 0),
                state.get('stocks_owned', 0),
                state.get('stocks_value', 0),
                state.get('portfolio_value', 0)
            ])
            state_vector = np.concatenate([price_history, portfolio_features])
        elif isinstance(state, np.ndarray):
            state_vector = state
        else:
            state_vector = np.array(state)
            
        # Ensure correct dimensions
        if len(state_vector) < self.state_dim:
            padding = np.zeros(self.state_dim - len(state_vector))
            state_vector = np.concatenate([state_vector, padding])
        elif len(state_vector) > self.state_dim:
            state_vector = state_vector[:self.state_dim]
            
        return state_vector

    def forward(self, state):
        """Forward pass through both actor and critic networks."""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(self.process_state(state)).to(self.device)
            
        actor_features = self.actor(state)
        action_mean = self.mean_layer(actor_features)
        action_log_std = self.log_std_layer(actor_features)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        
        value = self.critic(state)
        
        return action_mean, action_log_std, value
    
    def get_action(self, state, deterministic=False):
        """Sample an action from the policy given the current state."""
        state_tensor = torch.FloatTensor(self.process_state(state)).to(self.device)
        
        with torch.no_grad():
            action_mean, action_log_std, _ = self.forward(state_tensor)
            
            if deterministic:
                action = action_mean
            else:
                current_exploration = max(
                    self.min_exploration,
                    self.exploration_noise * (self.exploration_decay ** self.training_steps)
                )
                
                action_std = torch.exp(action_log_std) + current_exploration
                normal = Normal(action_mean, action_std)
                action = normal.sample()
                
                if np.random.random() < current_exploration * 0.3:
                    forced_action_type = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                    action[0] = forced_action_type
                    action[1] = 0.3 + np.random.random() * 0.7
            
            action_np = action.cpu().numpy()
            action_np[0] = np.clip(action_np[0], 0, 2)
            action_type = round(float(action_np[0]))
            action_np[0] = action_type
            action_np[1] = np.clip(action_np[1], 0, 1)
            
            self.action_counts[action_type] = self.action_counts.get(action_type, 0) + 1
            
            return action_np
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition for training."""
        self.states.append(self.process_state(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(self.process_state(next_state))
        self.dones.append(done)
        
    def update_policy(self):
        """Update the policy using stored transitions."""
        if len(self.states) < 1:
            return
            
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        
        # Get values
        _, _, values = self.forward(states)
        _, _, next_values = self.forward(next_states)
        
        # Calculate advantages
        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        
        # Calculate actor loss
        action_mean, action_log_std, _ = self.forward(states)
        action_std = torch.exp(action_log_std)
        normal = Normal(action_mean, action_std)
        log_probs = normal.log_prob(actions).sum(dim=1)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Calculate critic loss
        critic_loss = advantages.pow(2).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        # Update training steps
        self.training_steps += 1
        
    def save(self, path):
        """Save the model to a file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'mean_layer_state_dict': self.mean_layer.state_dict(),
            'log_std_layer_state_dict': self.log_std_layer.state_dict(),
            'training_steps': self.training_steps,
            'action_counts': self.action_counts
        }, path)
        
    def load(self, path):
        """Load the model from a file."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.mean_layer.load_state_dict(checkpoint['mean_layer_state_dict'])
        self.log_std_layer.load_state_dict(checkpoint['log_std_layer_state_dict'])
        self.training_steps = checkpoint.get('training_steps', 0)
        self.action_counts = checkpoint.get('action_counts', {0: 0, 1: 0, 2: 0}) 