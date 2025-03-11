"""
Price Simulator Module

This module provides tools for simulating stock price movements using geometric Brownian motion
and calculating option prices using the Black-Scholes model. It can be used for:
- Simulating realistic stock price paths
- Calculating theoretical option prices for different strikes and maturities
- Visualizing stock price movement scenarios
"""


import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class PriceSimulator:
    def __init__(self, initial_price, mu, sigma, risk_free_rate=0.02):
        """
        Initialize the stock price simulator.
        
        Parameters:
        initial_price (float): Initial stock price
        mu (float): Expected annual return (drift)
        sigma (float): Annual volatility
        risk_free_rate (float): Risk-free interest rate (annual)
        """
        self.initial_price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.risk_free_rate = risk_free_rate
        self.simulated_paths = None
        self.simulated_prices = None
        self.dates = None
        
    def simulate_path(self, days, num_simulations=1, seed=None):
        """
        Simulate stock price path using Geometric Brownian Motion.
        
        Parameters:
        days (int): Number of days to simulate
        num_simulations (int): Number of simulation paths
        seed (int): Random seed for reproducibility
        
        Returns:
        numpy.ndarray: Simulated price paths of shape (days+1, num_simulations)
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Daily parameters
        dt = 1/252  # Trading days in a year
        daily_drift = self.mu * dt
        daily_vol = self.sigma * np.sqrt(dt)
        
        # Generate random shocks
        Z = np.random.normal(0, 1, size=(days, num_simulations))
        
        # Initialize price array (including the initial price)
        prices = np.zeros((days + 1, num_simulations))
        prices[0] = self.initial_price
        
        # Simulate paths
        for t in range(1, days + 1):
            prices[t] = prices[t-1] * np.exp(daily_drift - 0.5 * daily_vol**2 + daily_vol * Z[t-1])
        
        self.simulated_paths = prices
        
        # Create dates (assuming we start from today)
        start_date = datetime.now()
        self.dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]
        
        # Store the mean path as the simulated price series
        self.simulated_prices = pd.Series(prices.mean(axis=1), index=self.dates)
        
        return prices
    
    def get_price_dataframe(self):
        """
        Returns a DataFrame with dates and simulated prices.
        """
        if self.simulated_prices is None:
            raise ValueError("Run simulate_path first")
            
        return pd.DataFrame({'Date': self.dates, 'Price': self.simulated_prices.values})
    
    def get_price_history(self, timestep:int):
        return self.simulated_prices[:timestep+1]
    
    def plot_simulation(self, title="Stock Price Simulation", show_individual_paths=False):
        """
        Plot the simulated stock price paths.
        
        Parameters:
        title (str): Plot title
        show_individual_paths (bool): Whether to show all individual simulation paths
        """
        if self.simulated_paths is None:
            raise ValueError("Run simulate_path first")
            
        plt.figure(figsize=(12, 6))
        
        if show_individual_paths:
            for i in range(self.simulated_paths.shape[1]):
                plt.plot(self.dates, self.simulated_paths[:, i], 'lightgray', alpha=0.3)
                
        plt.plot(self.dates, self.simulated_prices, 'b', linewidth=2, label='Mean Path')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.grid(True)
        return plt
    
    def black_scholes_call(self, strike, maturity_days):
        """
        Calculate Black-Scholes price for a call option.
        
        Parameters:
        strike (float): Strike price
        maturity_days (int): Days until option maturity
        
        Returns:
        float: Call option price
        """
        if self.simulated_prices is None:
            raise ValueError("Run simulate_path first")
            
        # Current stock price (S)
        S = self.simulated_prices[0]
        
        # Strike price (K)
        K = strike
        
        # Time to maturity in years
        T = maturity_days / 252
        
        # Risk-free rate (r)
        r = self.risk_free_rate
        
        # Volatility (σ)
        sigma = self.sigma
        
        # Black-Scholes formula components
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Call option price
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return call_price
    
    def black_scholes_put(self, strike, maturity_days):
        """
        Calculate Black-Scholes price for a put option.
        
        Parameters:
        strike (float): Strike price
        maturity_days (int): Days until option maturity
        
        Returns:
        float: Put option price
        """
        if self.simulated_prices is None:
            raise ValueError("Run simulate_path first")
            
        # Current stock price (S)
        S = self.simulated_prices[0]
        
        # Strike price (K)
        K = strike
        
        # Time to maturity in years
        T = maturity_days / 252
        
        # Risk-free rate (r)
        r = self.risk_free_rate
        
        # Volatility (σ)
        sigma = self.sigma
        
        # Black-Scholes formula components
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Put option price using put-call parity
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return put_price
    
    def calculate_option_prices(self, strikes, maturities):
        """
        Calculate option prices for various strikes and maturities.
        
        Parameters:
        strikes (list): List of strike prices
        maturities (list): List of maturities in days
        
        Returns:
        pandas.DataFrame: DataFrame with option prices
        """
        if self.simulated_prices is None:
            raise ValueError("Run simulate_path first")
            
        results = []
        
        for strike in strikes:
            for maturity in maturities:
                call_price = self.black_scholes_call(strike, maturity)
                put_price = self.black_scholes_put(strike, maturity)
                
                results.append({
                    'Strike': strike,
                    'Maturity (days)': maturity,
                    'Maturity Date': (datetime.now() + timedelta(days=maturity)).strftime('%Y-%m-%d'),
                    'Call Price': round(call_price, 2),
                    'Put Price': round(put_price, 2)
                })
                
        return pd.DataFrame(results)
