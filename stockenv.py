import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from price_simulator import PriceSimulator
from portfolio_class import Portfolio

class StockTradingEnv(gym.Env):
    """
    Custom Gymnasium environment for stock trading with options.
    
    This environment simulates a stock trading scenario where an agent can:
    - Buy/sell stocks
    - Buy/sell options (calls/puts)
    - Hold current position
    
    The state space includes price history, portfolio composition, and other relevant financial metrics.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(
        self, 
        initial_cash: float = 10000.0,
        initial_stock_price: float = 100.0,
        mu: float = 0.1,  # Expected annual return
        sigma: float = 0.2,  # Annual volatility
        risk_free_rate: float = 0.02,
        max_steps: int = 252,  # One trading year
        history_length: int = 10,  # Length of price history in state
        render_mode: Optional[str] = None,
        price_simulator: Optional[PriceSimulator] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            initial_cash: Starting cash amount
            initial_stock_price: Starting price of the stock
            mu: Expected annual return (drift)
            sigma: Annual volatility
            risk_free_rate: Risk-free interest rate
            max_steps: Maximum number of steps in an episode
            history_length: Number of days of price history to include in state
            render_mode: Mode for rendering the environment
            price_simulator: Optional pre-configured price simulator
            seed: Random seed
        """
        super().__init__()
        
        self.initial_cash = initial_cash
        self.initial_stock_price = initial_stock_price
        self.mu = mu
        self.sigma = sigma
        self.risk_free_rate = risk_free_rate
        self.max_steps = max_steps
        self.history_length = history_length
        self.render_mode = render_mode
        self.seed_value = seed
        self.history = {
                    'dates': [],
                    'portfolio_values': [],
                    'cash_values': [],
                    'stock_values': [],
                    'options_values': [],
                    'stock_prices': []
                }

        # Initialize environment components
        self._initialize_simulator(price_simulator)
        self.reset()
        
        # Action space: [action_type, amount]
        # action_type: 0 (buy), 1 (sell), 2 (hold)
        # amount: Proportion of available cash/stock to use (0.0 to 1.0)
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([2, 1]),
            dtype=np.float32
        )
        
        # State space
        # 1. Price history (history_length days)
        # 2. Portfolio info (cash, stock value, option value, total value)
        # 3. Holdings (current stock quantity)
        # 4. Return metrics (daily returns for history_length days)
        
        # Calculate observation space bounds
        price_history_dim = self.history_length
        portfolio_info_dim = 4  # cash, stock value, option value, total value
        holdings_dim = 1  # stock quantity
        returns_dim = self.history_length  # daily returns
        
        obs_dim = price_history_dim + portfolio_info_dim + holdings_dim + returns_dim
        
        # Wide bounds for all observation values
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def _initialize_simulator(self, price_simulator=None):
        """Initialize or use the provided price simulator."""
        if price_simulator is not None:
            self.price_simulator = price_simulator
        else:
            self.price_simulator = PriceSimulator(
                self.initial_stock_price, 
                self.mu, 
                self.sigma, 
                self.risk_free_rate
            )
            # Simulate the entire price path for the max_steps
            self.price_simulator.simulate_path(
                days=self.max_steps + self.history_length,
                seed=self.seed_value
            )
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options for resetting
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        if seed is not None and seed != self.seed_value:
            self.seed_value = seed
            self._initialize_simulator()
        
        # Reset time step
        self.current_step = 0
        
        # Generate start date (using the history length offset to have enough history)
        self.start_date = datetime.now()
        # self.start_date = self.start_date.strftime("%Y-%m-%d")
        self.current_date = (self.start_date + timedelta(days=self.history_length + self.current_step)).strftime("%Y-%m-%d")
        self.current_date_dt = self.start_date + timedelta(days=self.history_length + self.current_step)

        
        # Initialize portfolio
        self.portfolio = Portfolio(self.initial_cash)
        
        # Calculate initial portfolio value to track performance
        self.initial_portfolio_value = self.portfolio.get_portfolio_value(
            self.price_simulator, 
            self.current_date_dt
        )
        
        # Get the initial observation
        observation = self._get_observation()
        
        # Initialize episode info
        info = {
            "portfolio_value": self.initial_portfolio_value,
            "cash": self.portfolio.cash,
            "step": self.current_step,
            "date": self.current_date
        }

        self.history = {
                    'dates': [],
                    'portfolio_values': [],
                    'cash_values': [],
                    'stock_values': [],
                    'options_values': [],
                    'stock_prices': []
                }
        
        return observation, info
    
    def _get_observation(self):
        """
        Construct the observation array from the current state.
        
        Returns:
            numpy.ndarray: The observation vector
        """
        # Get current date and a range of historical dates
        date_index = self.history_length + self.current_step
        
        # 1. Get price history (normalized by current price)
        current_price = self.price_simulator.simulated_prices[date_index]
        history_range = range(date_index - self.history_length, date_index)
        price_history = self.price_simulator.simulated_prices[history_range]
        normalized_price_history = price_history / current_price
        
        # 2. Get portfolio information
        portfolio_summary = self.portfolio.get_portfolio_summary(
            self.price_simulator, 
            self.current_date_dt
        )
        
        cash = portfolio_summary["Cash"]
        stock_value = portfolio_summary["Stock Value"]
        options_value = portfolio_summary["Options Value"]
        total_value = portfolio_summary["Total Value"]
        
        # Normalize portfolio values by initial portfolio value
        normalized_portfolio_info = np.array([
            cash / self.initial_portfolio_value,
            stock_value / self.initial_portfolio_value,
            options_value / self.initial_portfolio_value,
            total_value / self.initial_portfolio_value
        ])
        
        # 3. Current holdings
        stock_quantity = portfolio_summary["Stock Quantity"]
        normalized_holdings = np.array([stock_quantity * current_price / self.initial_portfolio_value])
        
        # 4. Calculate returns
        returns = np.diff(self.price_simulator.simulated_prices[history_range]) / \
                 self.price_simulator.simulated_prices[history_range[:-1]]
        
        # Ensure returns array has expected length
        if len(returns) < self.history_length:
            padding = np.zeros(self.history_length - len(returns))
            returns = np.concatenate((padding, returns))
        
        # Combine all parts into one observation array
        observation = np.concatenate([
            normalized_price_history,
            normalized_portfolio_info,
            normalized_holdings,
            returns
        ]).astype(np.float32)
        
        return observation
    
    def _map_action(self, action):
        """
        Map the action from the action space to concrete trading actions.
        
        Args:
            action: The action from the action space (array)
            
        Returns:
            Tuple: (action_type, amount)
        """
        action_type = round(float(action[0]))  # 0: buy, 1: sell, 2: hold
        amount_ratio = max(0.0, min(1.0, float(action[1])))  # Clamp between 0 and 1
        
        # Determine actual amount based on the action type
        portfolio_summary = self.portfolio.get_portfolio_summary(
            self.price_simulator, 
            self.current_date_dt
        )
        
        if action_type == 0:  # Buy
            # Amount to buy based on available cash
            cash = portfolio_summary["Cash"]
            max_buy_amount = cash / self.price_simulator.simulated_prices[self.history_length + self.current_step]
            amount = int(max_buy_amount * amount_ratio)
        
        elif action_type == 1:  # Sell
            # Amount to sell based on current stock holdings
            current_holdings = portfolio_summary["Stock Quantity"]
            amount = int(current_holdings * amount_ratio)
        
        else:  # Hold
            amount = 0
        
        return action_type, amount
    
    def step(self, action):
        """
        Take a step in the environment using the given action.
        
        Args:
            action: Action to take (from action space)
            
        Returns:
            Tuple: (observation, reward, terminated, truncated, info)
        """
        # Map action to concrete trading decision
        action_type, amount = self._map_action(action)
        
        # Record portfolio value before action
        portfolio_value_before = self.portfolio.get_portfolio_value(
            self.price_simulator, 
            self.current_date_dt
        )
        
        # Execute action
        success = False
        if action_type == 0 and amount > 0:  # Buy
            success = self.portfolio.buy_stock(
                self.price_simulator, 
                self.current_date_dt, 
                amount
            )
        
        elif action_type == 1 and amount > 0:  # Sell
            success = self.portfolio.sell_stock(
                self.price_simulator, 
                self.current_date_dt, 
                amount
            )
        
        else:  # Hold or invalid action
            success = True  # Holding is always successful
        
        # Advance time step
        self.current_step += 1
        # self.current_date += timedelta(days=1)
        self.current_date = (self.start_date + timedelta(days=self.history_length + self.current_step)).strftime("%Y-%m-%d")
        print(f"{self.current_date} -> current date")
        self.current_date_dt = self.start_date + timedelta(days=self.history_length + self.current_step)
        
        # Get portfolio value after action
        portfolio_value_after = self.portfolio.get_portfolio_value(
            self.price_simulator, 
            self.current_date_dt
        )
        
        # Calculate reward (change in portfolio value)
        reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        
        # Add transaction cost penalty for unsuccessful actions
        if not success:
            reward -= 1  # Small penalty for failed transactions
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get the new observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            "portfolio_value": portfolio_value_after,
            "cash": self.portfolio.cash,
            "step": self.current_step,
            "date": self.current_date,
            "action_success": success,
            "action_type": ["buy", "sell", "hold"][action_type],
            "action_amount": amount,
            "stock_price": self.price_simulator.simulated_prices[self.history_length + self.current_step],
            "portfolio_return": portfolio_value_after / self.initial_portfolio_value - 1
        }
        
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Currently supports text-based rendering to console.
        Could be extended to produce visual rendering using matplotlib.
        
        Returns:
            Rendering of the environment state
        """
        portfolio_summary = self.portfolio.get_portfolio_summary(
                self.price_simulator, 
                self.current_date_dt
            )
        current_price = self.price_simulator.simulated_prices[self.history_length + self.current_step]
            
        # Print summary
        print(f"\n===== Step {self.current_step} | Date: {self.current_date} =====")
        print(f"Stock Price: ${current_price:.2f}")
        print(f"Cash: ${portfolio_summary['Cash']:.2f}")
        print(f"Stock Value: ${portfolio_summary['Stock Value']:.2f}")
        print(f"Stock Quantity: {portfolio_summary['Stock Quantity']}")
        print(f"Options Value: ${portfolio_summary['Options Value']:.2f}")
        print(f"Total Portfolio Value: ${portfolio_summary['Total Value']:.2f}")
        print(f"Return since start: {portfolio_summary['Total Value']/self.initial_portfolio_value - 1:.2%}")
        print("=" * 50)

        self.history['dates'].append(self.current_date)
        self.history['portfolio_values'].append(portfolio_summary['Total Value'])
        self.history['cash_values'].append(portfolio_summary['Cash'])
        self.history['stock_values'].append(portfolio_summary['Stock Value'])
        self.history['options_values'].append(portfolio_summary['Options Value'])
        self.history['stock_prices'].append(current_price)
        return self.history

    def close(self):
        """
        Clean up environment resources.
        """
        pass


# Example usage with a basic random agent
if __name__ == "__main__":
    # Create the environment
    env = StockTradingEnv(
        initial_cash=10000.0,
        initial_stock_price=100.0,
        mu=0.1,
        sigma=0.2,
        max_steps=100,
        render_mode="human"
    )
    
    # Reset the environment
    observation, info = env.reset(seed=42)
    
    # Run a simple random agent
    total_reward = 0
    for _ in range(100):
        # Sample a random action
        action = env.action_space.sample()
        
        # Take the action in the environment
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render the environment
        env.render()
        
        if terminated or truncated:
            print(f"Episode finished with total reward: {total_reward}")
            break
    
    env.close()