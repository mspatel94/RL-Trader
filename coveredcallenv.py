
# covered_call_trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from price_simulator import PriceSimulator
from covered_portfolio import Portfolio
from typing import Optional

class CoveredCallTradingEnv(gym.Env):
    """
    Gymnasium environment for trading stocks with covered calls.
    Allowed actions:
      0: BUY_STOCK
      1: SELL_STOCK
      2: HOLD
      3: SELL_COVERED_CALL (only if sufficient underlying stock exists)
      4: CLOSE_COVERED_CALL (buy to close a short call position)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
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
        seed: Optional[int] = None,
        start_date: Optional[datetime] = None
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
        self.start_date = datetime.now()  # Use a fixed start date instead of now
        self.history = {
            'dates': [],
            'portfolio_values': [],
            'cash_values': [],
            'stock_values': [],
            'options_values': [],
            'stock_prices': []
        }
        
        self._initialize_simulator(price_simulator)
        self.reset()
        
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([4, 1]),
            dtype=np.float32
        )
        
        # Observation space: price history + portfolio info + stock holdings + returns + call option holdings
        price_history_dim = self.history_length
        portfolio_info_dim = 4  # cash, stock value, options value, total value
        stock_holdings_dim = 1
        option_holdings_dim = 1   # only call options
        returns_dim = self.history_length
        obs_dim = price_history_dim + portfolio_info_dim + stock_holdings_dim + option_holdings_dim + returns_dim
        
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
        self.current_step = 0
        
        self.current_date = (self.start_date + timedelta(days=self.history_length + self.current_step)).strftime("%Y-%m-%d")
        self.current_date_dt = self.start_date + timedelta(days=self.history_length + self.current_step)
        
        self.portfolio = Portfolio(self.initial_cash)
        self.initial_portfolio_value = self.portfolio.get_portfolio_value(self.price_simulator, self.current_date_dt)
        
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
        current_price = self.price_simulator.simulated_prices.iloc[date_index]
        history_range = range(date_index - self.history_length, date_index)
        price_history = self.price_simulator.simulated_prices.iloc[history_range]
        normalized_price_history = price_history / current_price
        
        portfolio_summary = self.portfolio.get_portfolio_summary(self.price_simulator, self.current_date_dt)
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
        
        returns = np.diff(self.price_simulator.simulated_prices.iloc[history_range]) / \
                  self.price_simulator.simulated_prices.iloc[list(history_range)[:-1]]
        if len(returns) < self.history_length:
            padding = np.zeros(self.history_length - len(returns))
            returns = np.concatenate((padding, returns))
        
        call_qty = portfolio_summary["Call Options Quantity"]
        
        observation = np.concatenate([
            normalized_price_history,
            normalized_portfolio_info,
            normalized_holdings,
            returns,
            np.array([call_qty])
        ]).astype(np.float32)
        
        return observation
    
    def _map_action(self, action):
        """
        Map the action vector to one of the five allowed trading commands.
        """
        raw_type = int(round(float(action[0])))
        action_type = max(0, min(4, raw_type))
        amount_ratio = max(0.0, min(1.0, float(action[1])))
        
        date_index = self.history_length + self.current_step
        current_price = self.price_simulator.simulated_prices.iloc[date_index]
        
        if action_type == 0:  # BUY_STOCK
            portfolio_summary = self.portfolio.get_portfolio_summary(self.price_simulator, self.current_date_dt)
            cash_available = portfolio_summary["Cash"]
            max_shares = int(cash_available // current_price)
            shares_to_buy = int(max_shares * amount_ratio)
            return ("BUY_STOCK", shares_to_buy)
        elif action_type == 1:  
            portfolio_summary = self.portfolio.get_portfolio_summary(self.price_simulator, self.current_date_dt)
            stock_quantity = portfolio_summary["Stock Quantity"]
            shares_to_sell = int(stock_quantity * amount_ratio)
            return ("SELL_STOCK", shares_to_sell)
        elif action_type == 2:  
            return ("HOLD", 0)
        elif action_type == 3:  
            # Use a fixed 30-day maturity and strike equal to current stock price
            maturity_dt = self.current_date_dt + timedelta(days=30)  
            strike_price = float(current_price)
            portfolio_summary = self.portfolio.get_portfolio_summary(self.price_simulator, self.current_date_dt)
            available_shares = portfolio_summary["Stock Quantity"]
            max_contracts = available_shares  
            contracts_to_sell = int(max_contracts * amount_ratio)
            return ("SELL_COVERED_CALL", (strike_price, maturity_dt, contracts_to_sell))
        elif action_type == 4:  # CLOSE_COVERED_CALL
            open_calls = []
            for option_id, quantity in self.portfolio.current_holdings.items():
                if option_id.startswith("CALL_"):
                    if quantity < 0: 
                        open_calls.append((option_id, quantity))

            if not open_calls:
                return ("HOLD", 0)
            option_id, current_qty = open_calls[0]
            option_type, strike, maturity_str = self.portfolio._parse_option_id(option_id)
            maturity_dt = datetime.strptime(maturity_str, "%Y-%m-%d")
            contracts_to_close = int(abs(current_qty) * amount_ratio)
            return ("CLOSE_COVERED_CALL", (float(strike), maturity_dt, contracts_to_close))
        return ("HOLD", 0)
    
    def step(self, action):
        """
        Execute one time step within the environment.
        """
        command, data = self._map_action(action)
        portfolio_value_before = self.portfolio.get_portfolio_value(self.price_simulator, self.current_date_dt)
        
        self.portfolio.close_covered_calls_expiring_tomorrow(self.price_simulator, self.current_date_dt)
        
        success = True
        if command == "BUY_STOCK":
            shares_to_buy = data
            if shares_to_buy > 0:
                success = self.portfolio.buy_stock(self.price_simulator, self.current_date_dt, shares_to_buy)
        elif command == "SELL_STOCK":
            shares_to_sell = data
            if shares_to_sell > 0:
                success = self.portfolio.sell_stock(self.price_simulator, self.current_date_dt, shares_to_sell)
        elif command == "SELL_COVERED_CALL":
            strike, maturity_dt, qty = data
            if qty > 0 and maturity_dt > self.current_date_dt:
                success = self.portfolio.sell_covered_call(self.price_simulator, self.current_date_dt, strike, maturity_dt, qty)
        elif command == "CLOSE_COVERED_CALL":
            strike, maturity_dt, qty = data
            if qty > 0 and maturity_dt > self.current_date_dt:
                success = self.portfolio.buy_option(self.price_simulator, self.current_date_dt, "CALL", strike, maturity_dt, qty)
        
        self.current_step += 1
        self.current_date = (self.start_date + timedelta(days=self.history_length + self.current_step)).strftime("%Y-%m-%d")
        self.current_date_dt = self.start_date + timedelta(days=self.history_length + self.current_step)
        
        portfolio_value_after = self.portfolio.get_portfolio_value(self.price_simulator, self.current_date_dt)
        reward = 100*(portfolio_value_after - portfolio_value_before) / portfolio_value_before

        if not success:
            reward -= 0.001
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        observation = self._get_observation()
        info = {
            "portfolio_value": portfolio_value_after,
            "cash": self.portfolio.cash,
            "step": self.current_step,
            "date": self.current_date,
            "action_success": success,
            "action_type": command,
            "action_amount": data,
            "stock_price": self.price_simulator.simulated_prices.iloc[self.history_length + self.current_step],
            "portfolio_return": portfolio_value_after / self.initial_portfolio_value - 1
        }

        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render environment state to the console.
        """
        portfolio_summary = self.portfolio.get_portfolio_summary(self.price_simulator, self.current_date_dt)
        current_price = self.price_simulator.simulated_prices.iloc[self.history_length + self.current_step]
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
        pass
