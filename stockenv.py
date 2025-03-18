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
    Custom Gymnasium environment for stock trading only.
    
    
    """    
    def __init__(
        self, 
        initial_cash: float = 10000.0,
        initial_stock_price: float = 100.0,
        mu: float = 0.1,  
        sigma: float = 0.2,
        risk_free_rate: float = 0.02,
        max_steps: int = 252,
        history_length: int = 10,
        render_mode: Optional[str] = None,
        price_simulator: Optional[PriceSimulator] = None,
        seed: Optional[int] = None
    ):
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
        
        
        price_history_dim = self.history_length
        portfolio_info_dim = 4  
        holdings_dim = 1 
        returns_dim = self.history_length 
        
        obs_dim = price_history_dim + portfolio_info_dim + holdings_dim + returns_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def _initialize_simulator(self, price_simulator=None):
        if price_simulator is not None:
            self.price_simulator = price_simulator
        else:
            self.price_simulator = PriceSimulator(
                self.initial_stock_price, 
                self.mu, 
                self.sigma, 
                self.risk_free_rate
            )
            self.price_simulator.simulate_path(
                days=self.max_steps + self.history_length,
                seed=self.seed_value
            )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None and seed != self.seed_value:
            self.seed_value = seed
            self._initialize_simulator()
        
        # Reset time step
        self.current_step = 0
        
        self.start_date = datetime.now()
        self.current_date = (self.start_date + timedelta(days=self.history_length + self.current_step)).strftime("%Y-%m-%d")
        self.current_date_dt = self.start_date + timedelta(days=self.history_length + self.current_step)

        
        self.portfolio = Portfolio(self.initial_cash)
        
        self.initial_portfolio_value = self.portfolio.get_portfolio_value(
            self.price_simulator, 
            self.current_date_dt
        )
        
        observation = self._get_observation()
        
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
        
        date_index = self.history_length + self.current_step
        
        
        current_price = self.price_simulator.simulated_prices[date_index]
        history_range = range(date_index - self.history_length, date_index)
        price_history = self.price_simulator.simulated_prices[history_range]
        normalized_price_history = price_history / current_price
        portfolio_summary = self.portfolio.get_portfolio_summary(
            self.price_simulator, 
            self.current_date_dt
        )
        
        cash = portfolio_summary["Cash"]
        stock_value = portfolio_summary["Stock Value"]
        options_value = portfolio_summary["Options Value"]
        total_value = portfolio_summary["Total Value"]
        
        normalized_portfolio_info = np.array([
            cash / self.initial_portfolio_value,
            stock_value / self.initial_portfolio_value,
            options_value / self.initial_portfolio_value,
            total_value / self.initial_portfolio_value
        ])
        

        stock_quantity = portfolio_summary["Stock Quantity"]
        normalized_holdings = np.array([stock_quantity * current_price / self.initial_portfolio_value])
        
        
        returns = np.diff(self.price_simulator.simulated_prices[history_range]) / \
                 self.price_simulator.simulated_prices[history_range[:-1]]
        
        if len(returns) < self.history_length:
            padding = np.zeros(self.history_length - len(returns))
            returns = np.concatenate((padding, returns))
        
        observation = np.concatenate([
            normalized_price_history,
            normalized_portfolio_info,
            normalized_holdings,
            returns
        ]).astype(np.float32)
        
        return observation
    
    def _map_action(self, action):
        action_type = round(float(action[0]))  
        amount_ratio = max(0.0, min(1.0, float(action[1]))) 
        
        portfolio_summary = self.portfolio.get_portfolio_summary(
            self.price_simulator, 
            self.current_date_dt
        )
        
        if action_type == 0:  
            cash = portfolio_summary["Cash"]
            max_buy_amount = cash / self.price_simulator.simulated_prices[self.history_length + self.current_step]
            amount = int(max_buy_amount * amount_ratio)
        elif action_type == 1:  
            current_holdings = portfolio_summary["Stock Quantity"]
            amount = int(current_holdings * amount_ratio)
        else:
            amount = 0
        
        return action_type, amount
    
    def step(self, action):
        action_type, amount = self._map_action(action)
        
        portfolio_value_before = self.portfolio.get_portfolio_value(
            self.price_simulator, 
            self.current_date_dt
        )
        
        success = False
        if action_type == 0 and amount > 0:  
            success = self.portfolio.buy_stock(
                self.price_simulator, 
                self.current_date_dt, 
                amount
            )
        
        elif action_type == 1 and amount > 0: 
            success = self.portfolio.sell_stock(
                self.price_simulator, 
                self.current_date_dt, 
                amount
            )
        
        else:  
            success = True  
        
        self.current_step += 1
        self.current_date = (self.start_date + timedelta(days=self.history_length + self.current_step)).strftime("%Y-%m-%d")
        self.current_date_dt = self.start_date + timedelta(days=self.history_length + self.current_step)
        
        portfolio_value_after = self.portfolio.get_portfolio_value(
            self.price_simulator, 
            self.current_date_dt
        )
        
        reward = (portfolio_value_after - portfolio_value_before) / portfolio_value_before
        
        if not success:
            reward -= 1  
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        observation = self._get_observation()
        
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
        portfolio_summary = self.portfolio.get_portfolio_summary(
                self.price_simulator, 
                self.current_date_dt
            )
        current_price = self.price_simulator.simulated_prices[self.history_length + self.current_step]
            
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

