from typing import List

class Cash:
    def __init__(self, amount: float):
        self.amount = amount

    def __repr__(self):
        return f"Cash: {self.amount}"

class Stock:
    def __init__(self, symbol: str, price: int, history: List[int]):
        self.symbol = symbol
        self.price = price
        self.history = history

#TODO(farzad): add relevant bits for options
class Option:
    def __init__(self, stock:Stock):
        self.stock = stock