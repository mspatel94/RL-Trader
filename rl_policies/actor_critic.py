"""
Actor-Critic Reinforcement Learning Policy for trading options.
Combines policy-based and value-based methods where the Actor decides actions
and the Critic evaluates them by computing the value function.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network with shared feature extraction layers.
    
    The network takes a state as input and outputs:
    1. Action probabilities (Actor)
    2. State value (Critic)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the Actor-Critic network.
        
        Parameters:
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
        hidden_dim (int): Dimension of the hidden layers
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value network)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        """
        Forward pass through the network.
        
        Parameters:
        state (torch.Tensor): Current state
        
        Returns:
        tuple: (action_probs, state_value)
        """
        features = self.shared_layers(state)
        action_probs = self.actor_head(features)
        state_value = self.critic_head(features)
        
        return action_probs, state_value
    
    def get_action(self, state, device='cpu'):
        """
        Sample an action from the policy.
        
        Parameters:
        state (numpy.ndarray): Current state
        device (str): Device to run the model on
        
        Returns:
        tuple: (action, log_prob, entropy)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs, _ = self.forward(state_tensor)
        
        # Create a categorical distribution over the action probabilities
        dist = Categorical(action_probs)
        
        # Sample an action
        action = dist.sample()
        
        # Calculate log probability and entropy
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.item(), log_prob, entropy


class ActorCriticAgent:
    """
    Actor-Critic agent for trading options.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.99, device='cpu'):
        """
        Initialize the Actor-Critic agent.
        
        Parameters:
        state_dim (int): Dimension of the state space
        action_dim (int): Dimension of the action space
        hidden_dim (int): Dimension of the hidden layers
        lr (float): Learning rate
        gamma (float): Discount factor
        device (str): Device to run the model on
        """
        self.device = device
        self.gamma = gamma
        
        # Initialize the Actor-Critic network
        self.network = ActorCriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # Initialize the optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Initialize memory for storing episode data
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []
        
    def select_action(self, state):
        """
        Select an action based on the current state.
        
        Parameters:
        state (numpy.ndarray): Current state
        
        Returns:
        int: Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities and state value
        action_probs, state_value = self.network(state_tensor)
        
        # Create a categorical distribution over the action probabilities
        dist = Categorical(action_probs)
        
        # Sample an action
        action = dist.sample()
        
        # Store episode data
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(dist.log_prob(action))
        self.values.append(state_value)
        self.entropies.append(dist.entropy())
        
        return action.item()
    
    def update_policy(self, next_state=None, done=False):
        """
        Update the policy using the collected experience.
        
        Parameters:
        next_state (numpy.ndarray): Next state (None if episode is done)
        done (bool): Whether the episode is done
        
        Returns:
        float: Loss value
        """
        # If there are no experiences, return 0
        if len(self.rewards) == 0:
            return 0.0
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = 0
        
        # If the episode is not done, bootstrap with the value of the next state
        if not done and next_state is not None:
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            _, next_value = self.network(next_state_tensor)
            R = next_value.item()
        
        # Calculate returns and advantages
        for reward, value in zip(reversed(self.rewards), reversed(self.values)):
            R = reward + self.gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - value.item())
        
        # Convert lists to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        log_probs = torch.stack(self.log_probs)
        values = torch.cat(self.values)
        entropies = torch.stack(self.entropies)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate actor loss (policy gradient with advantage)
        actor_loss = -(log_probs * advantages).mean()
        
        # Calculate critic loss (MSE between predicted values and returns)
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Calculate entropy loss (to encourage exploration)
        entropy_loss = -entropies.mean()
        
        # Total loss
        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
        
        # Update the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []
        
        return loss.item()
    
    def store_reward(self, reward):
        """
        Store a reward.
        
        Parameters:
        reward (float): Reward
        """
        self.rewards.append(reward)
    
    def save_model(self, path):
        """
        Save the model.
        
        Parameters:
        path (str): Path to save the model
        """
        torch.save(self.network.state_dict(), path)
    
    def load_model(self, path):
        """
        Load the model.
        
        Parameters:
        path (str): Path to load the model from
        """
        self.network.load_state_dict(torch.load(path))


class TradingEnvironment:
    """
    Trading environment for options trading with actions:
    - Buy call options (1)
    - Buy put options (2)
    - Hold cash (0)
    """
    def __init__(self, price_simulator, initial_cash=10000, episode_length=252, window_size=10):
        """
        Initialize the trading environment.
        
        Parameters:
        price_simulator (PriceSimulator): Price simulator
        initial_cash (float): Initial cash
        episode_length (int): Length of an episode in days
        window_size (int): Size of the historical price window to include in the state
        """
        self.price_simulator = price_simulator
        self.initial_cash = initial_cash
        self.episode_length = episode_length
        self.window_size = window_size
        
        # Action space: 0 = hold cash, 1 = buy call, 2 = buy put
        self.action_space = 3
        
        # State space: current price, historical prices, portfolio value, time remaining
        self.state_dim = window_size + 2
        
        # Reset the environment
        self.reset()
    
    def reset(self):
        """
        Reset the environment.
        
        Returns:
        numpy.ndarray: Initial state
        """
        # Simulate stock prices for the episode
        self.price_simulator.simulate_path(self.episode_length)
        self.prices = self.price_simulator.simulated_paths[:, 0]  # Use the first simulation path
        
        # Initialize portfolio
        self.cash = self.initial_cash
        self.position = 0  # 0 = cash, 1 = call, 2 = put
        self.position_value = 0
        self.portfolio_value = self.cash
        self.portfolio_history = [self.portfolio_value]
        
        # Initialize time
        self.current_step = 0
        
        # Calculate initial state
        return self._get_state()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Parameters:
        action (int): Action to take (0 = hold cash, 1 = buy call, 2 = buy put)
        
        Returns:
        tuple: (next_state, reward, done, info)
        """
        # Get current price
        current_price = self.prices[self.current_step]
        
        # Close any existing position
        if self.position == 1:  # Call option
            # Calculate call option payoff
            strike = current_price * 0.95  # 5% out of the money
            next_price = self.prices[self.current_step + 1]
            payoff = max(0, next_price - strike)
            self.cash += payoff
        elif self.position == 2:  # Put option
            # Calculate put option payoff
            strike = current_price * 1.05  # 5% out of the money
            next_price = self.prices[self.current_step + 1]
            payoff = max(0, strike - next_price)
            self.cash += payoff
        
        # Take new position
        self.position = action
        
        # If buying an option, calculate the premium
        if action == 1:  # Call option
            strike = current_price * 0.95  # 5% out of the money
            premium = self.price_simulator.black_scholes_call(strike, 1)
            self.cash -= premium
        elif action == 2:  # Put option
            strike = current_price * 1.05  # 5% out of the money
            premium = self.price_simulator.black_scholes_put(strike, 1)
            self.cash -= premium
        
        # Update time
        self.current_step += 1
        
        # Calculate new portfolio value
        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate reward (change in portfolio value)
        reward = self.portfolio_value - old_portfolio_value
        
        # Check if episode is done
        done = self.current_step >= self.episode_length - 1
        
        # Get next state
        next_state = self._get_state()
        
        # Return step information
        info = {
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'position': self.position
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """
        Get the current state.
        
        Returns:
        numpy.ndarray: Current state
        """
        # Get historical prices
        if self.current_step < self.window_size:
            # Pad with initial price if not enough history
            price_history = [self.prices[0]] * (self.window_size - self.current_step) + \
                           list(self.prices[:self.current_step + 1])
        else:
            price_history = list(self.prices[self.current_step - self.window_size + 1:self.current_step + 1])
        
        # Normalize price history
        price_history = [p / price_history[-1] for p in price_history]
        
        # Combine state components
        state = price_history + [
            self.portfolio_value / self.initial_cash,  # Normalized portfolio value
            (self.episode_length - self.current_step) / self.episode_length  # Normalized time remaining
        ]
        
        return np.array(state)
    
    def render(self):
        """
        Render the environment.
        """
        plt.figure(figsize=(12, 6))
        
        # Plot stock price
        plt.subplot(2, 1, 1)
        plt.plot(self.prices[:self.current_step + 1])
        plt.title('Stock Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        
        # Plot portfolio value
        plt.subplot(2, 1, 2)
        plt.plot(self.portfolio_history)
        plt.title('Portfolio Value')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
        plt.tight_layout()
        plt.show()


def train_actor_critic(env, agent, num_episodes=1000, max_steps=None, render_interval=100):
    """
    Train an Actor-Critic agent.
    
    Parameters:
    env (TradingEnvironment): Trading environment
    agent (ActorCriticAgent): Actor-Critic agent
    num_episodes (int): Number of episodes to train for
    max_steps (int): Maximum number of steps per episode (None for no limit)
    render_interval (int): Interval at which to render the environment
    
    Returns:
    tuple: (episode_rewards, episode_lengths)
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store reward
            agent.store_reward(reward)
            
            # Update total reward
            total_reward += reward
            
            # Update state
            state = next_state
            
            # Increment step counter
            steps += 1
            
            # Check if maximum steps reached
            if max_steps is not None and steps >= max_steps:
                break
        
        # Update policy
        agent.update_policy(done=True)
        
        # Store episode statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Render environment
        if (episode + 1) % render_interval == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")
            env.render()
    
    return episode_rewards, episode_lengths


def evaluate_agent(env, agent, num_episodes=10, render=True):
    """
    Evaluate an Actor-Critic agent.
    
    Parameters:
    env (TradingEnvironment): Trading environment
    agent (ActorCriticAgent): Actor-Critic agent
    num_episodes (int): Number of episodes to evaluate for
    render (bool): Whether to render the environment
    
    Returns:
    float: Average reward
    """
    total_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Select action (without storing in memory)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action_probs, _ = agent.network(state_tensor)
            action = torch.argmax(action_probs).item()
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Update total reward
            total_reward += reward
            
            # Update state
            state = next_state
        
        # Store episode reward
        total_rewards.append(total_reward)
        
        # Render environment
        if render:
            print(f"Evaluation Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")
            env.render()
    
    # Calculate average reward
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"Average Reward: {avg_reward:.2f}")
    
    return avg_reward


if __name__ == "__main__":
    # Example usage
    from price_simulator.price_simulator import PriceSimulator
    
    # Initialize price simulator
    price_simulator = PriceSimulator(
        initial_price=100,
        mu=0.08,  # 8% annual return
        sigma=0.2,  # 20% annual volatility
        risk_free_rate=0.02  # 2% risk-free rate
    )
    
    # Initialize trading environment
    env = TradingEnvironment(
        price_simulator=price_simulator,
        initial_cash=10000,
        episode_length=252,  # 1 trading year
        window_size=10
    )
    
    # Initialize agent
    agent = ActorCriticAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space,
        hidden_dim=128,
        lr=0.001,
        gamma=0.99
    )
    
    # Train agent
    train_actor_critic(
        env=env,
        agent=agent,
        num_episodes=100,
        render_interval=20
    )
    
    # Evaluate agent
    evaluate_agent(
        env=env,
        agent=agent,
        num_episodes=5
    )
    
    # Save model
    agent.save_model("actor_critic_model.pt") 