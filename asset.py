from typing import List

class Cash:
    def __init__(self, amount: float):
        self.amount = amount

    def __repr__(self):
        return f"Cash: {self.amount}"

class Stock:
    def __init__(self, symbol: str, price: int, history: List[int], mu=0.1, sigma=0.1, risk_free_rate=0.03):
        """
        Parameters:
        initial_price (float): Initial stock price
        mu (float): Expected annual return (drift)
        sigma (float): Annual volatility
        risk_free_rate (float): Risk-free interest rate (annual)
        """

        self.symbol = symbol
        self.price = price
        self.history = history
        self.future_prices = []
        self._mu = mu
        self._sigma = sigma
        self.risk_free_rate = risk_free_rate

#TODO(farzad): add relevant bits for options
class Option:
    def __init__(self, stock:Stock):
        self.stock = stock
