import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from policy_interface import Policy
from portfolio import PortfolioState

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim * 2),  # For buy and sell quantities
        )
        
    def forward(self, state):
        output = self.network(state)
        # Split output into buy and sell actions
        buy_sell = output.reshape(-1, 2)
        # Apply softplus to ensure positive quantities
        return torch.nn.functional.softplus(buy_sell)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Outputs a single value
        )
        
    def forward(self, state):
        return self.network(state)

class ActorCriticPolicy(Policy):
    def __init__(self, state_dim=10, action_dim=1, lr=0.001, gamma=0.99):
        self.actor = ActorNetwork(state_dim, action_dim)
        self.critic = CriticNetwork(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
    def _extract_features(self, portfolio_state):
        """Extract relevant features from portfolio state."""
        if len(portfolio_state.history) == 0:
            # Return default values if history is empty
            return torch.zeros(10)
        
        # Get the latest portfolio
        portfolio = portfolio_state.history[-1]
        
        # For simplicity, we'll just use a few basic features:
        # - Current cash
        # - Current total value
        # - Some price history (if available)
        features = [portfolio.cash]
        
        # Add placeholder for portfolio value
        features.append(portfolio.get_value() if hasattr(portfolio, 'get_value') else 0)
        
        # Pad with zeros to reach state_dim
        while len(features) < 10:
            features.append(0)
            
        return torch.FloatTensor(features)
    
    def get_action(self, portfolio_state):
        """
        Implement the policy's action selection.
        Returns a tuple (buy_quantity, sell_quantity)
        """
        state_tensor = self._extract_features(portfolio_state)
        
        # Use the actor to predict actions
        with torch.no_grad():
            actions = self.actor(state_tensor)
        
        # Convert to numpy for easier handling
        buy_quantity, sell_quantity = actions[0].numpy()
        
        # Round to whole numbers for simplicity
        buy_quantity = round(buy_quantity)
        sell_quantity = round(sell_quantity)
        
        # Store state and action for training
        self.states.append(state_tensor)
        self.actions.append(torch.FloatTensor([buy_quantity, sell_quantity]))
        
        return buy_quantity, sell_quantity
    
    def update(self, reward, next_state, done=False):
        """
        Update the policy with the observed reward and next state.
        Should be called after get_action.
        """
        self.rewards.append(reward)
        self.next_states.append(self._extract_features(next_state))
        self.dones.append(done)
        
        # If episode is done, train on collected data
        if done:
            self._train()
            
    def _train(self):
        """Train the actor and critic networks using collected trajectories."""
        if len(self.states) == 0:
            return
            
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.FloatTensor(self.dones)
        
        # Calculate returns and advantages
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            target_values = rewards + self.gamma * next_values * (1 - dones)
            
        # Update critic
        current_values = self.critic(states).squeeze()
        critic_loss = nn.MSELoss()(current_values, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Calculate advantage
        advantages = target_values - current_values.detach()
        
        # Update actor
        log_probs = -0.5 * ((actions - self.actor(states)) ** 2).sum(dim=1)
        actor_loss = -(log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Reset trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = [] 