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
        if seed is not None:
            np.random.seed(seed)
            
        dt = 1/252  # Trading days in a year
        daily_drift = self.mu * dt
        daily_vol = self.sigma * np.sqrt(dt)
        
        Z = np.random.normal(0, 1, size=(days, num_simulations))
        
        prices = np.zeros((days + 1, num_simulations))
        prices[0] = self.initial_price
        
        for t in range(1, days + 1):
            prices[t] = prices[t-1] * np.exp(daily_drift - 0.5 * daily_vol**2 + daily_vol * Z[t-1])
        
        self.simulated_paths = prices
        
        start_date = datetime.now()
        self.dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]
        
        self.simulated_prices = pd.Series(prices.mean(axis=1), index=self.dates)
        
        return prices
    
    def get_price_dataframe(self):
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
    
    def black_scholes_call(self, strike, maturity_days, current_price=None, current_date=None):
        if current_price is None or current_date is None:
            raise ValueError("Provide current_price and current_date")

        if strike is None or maturity_days is None or strike==0 or maturity_days==0:
            raise ValueError("Provide strike and maturity_days")

        if self.simulated_prices is None and current_price is None:
            raise ValueError("Run simulate_path first or provide current_price")
        
        # Current stock price (S)
        S = current_price
        
        # Strike price (K)
        K = strike
        
        # Time to maturity in years
        if current_date is not None:
            maturity_date = current_date + timedelta(days=maturity_days)
            remaining_days = (maturity_date - current_date).days
            tau = max(0, remaining_days) / 252.0  # Ensure non-negative
        else:
            tau = maturity_days / 252.0
        
        if tau <= 0:
            raise ValueError("Time to maturity is negative")

        # Risk-free rate (r)
        r = self.risk_free_rate
        
        # Volatility (σ)
        sigma = self.sigma
        
        # Black-Scholes formula components
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        
        # Call option price
        call_price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        
        return call_price
    
    def black_scholes_put(self, strike, maturity_days, current_price=None, current_date=None):
        if self.simulated_prices is None and current_price is None:
            raise ValueError("Run simulate_path first or provide current_price")
            
        # Current stock price (S)
        S = current_price if current_price is not None else self.simulated_prices[0]
        
        # Strike price (K)
        K = strike

        print(f"Strike: {strike}, Maturity: {maturity_days}, Current Price: {current_price}, Current Date: {current_date}")
        
        # Time to maturity in years
        if current_date is not None:
            maturity_date = current_date + timedelta(days=maturity_days)
            remaining_days = (maturity_date - current_date).days
            tau = max(0, remaining_days) / 252  # Ensure non-negative
        else:
            tau = maturity_days / 252
        
        # Risk-free rate (r)
        r = self.risk_free_rate
        
        # Volatility (σ)
        sigma = self.sigma
        
        if tau <= 0:
            raise ValueError("Time to maturity is negative")
        
        # Black-Scholes formula components
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        
        # Put option price using put-call parity
        put_price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return put_price
    
    def calculate_option_prices(self, strikes, maturities, current_price=None, current_date=None):
        """
        Calculate option prices for various strikes and maturities.
        
        Parameters:
        strikes (list): List of strike prices
        maturities (list): List of maturities in days from time 0
        current_price (float, optional): Current stock price (uses initial price if None)
        current_date (datetime, optional): Current date for calculating remaining time to maturity
        
        Returns:
        pandas.DataFrame: DataFrame with option prices
        """
        if self.simulated_prices is None and current_price is None:
            raise ValueError("Run simulate_path first or provide current_price")
            
        results = []
        
        for strike in strikes:
            for maturity in maturities:
                call_price = self.black_scholes_call(strike, maturity, current_price, current_date)
                put_price = self.black_scholes_put(strike, maturity, current_price, current_date)
                
                # Calculate actual maturity date based on current date or simulation start
                if current_date is not None:
                    maturity_date = current_date + timedelta(days=maturity)
                else:
                    maturity_date = datetime.now() + timedelta(days=maturity)
                
                results.append({
                    'Strike': strike,
                    'Maturity (days)': maturity,
                    'Maturity Date': maturity_date.strftime('%Y-%m-%d'),
                    'Call Price': round(call_price, 2),
                    'Put Price': round(put_price, 2)
                })
                
        return pd.DataFrame(results)
