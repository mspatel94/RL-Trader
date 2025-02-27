#TODO: Add random variable that makes us sell the stock at each time step

class Portfolio:
    def __init__(self, cash_amount):
        self.cash = cash_amount
        self.stocks = {}
    
    def buy_stock(self, stock, quantity):
        raise NotImplementedError
    
    def sell_stock(self, stock, quantity):
        raise NotImplementedError
    
    def buy_option(self, option, quantity):
        raise NotImplementedError
    
    def sell_option(self, option, quantity):
        raise NotImplementedError

class PortfolioState:
    def __init__(self):
        self.history = []
        self.timestamps = []
    
    def add_state(self, state:Portfolio, timestamp):
        self.timestamps.append(timestamp)
        self.history.append(state)