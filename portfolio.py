#TODO: Add random variable that makes us sell the stock at each time step

class Portfolio:
    def __init__(self, cash_amount):
        self.cash = cash_amount
        self.stocks = {}
    
    def buy_stock(self, stock, quantity):
        //TODO(Farzad): start tracking prices in a data structure by calling price simulator
        raise NotImplementedError
    
    def sell_stock(self, stock, quantity):
        raise NotImplementedError
    
    def buy_to_close_option(self, option, quantity):
        raise NotImplementedError

    #Buy
    def sell_option(self, option, quantity):
        raise NotImplementedError

    

class PortfolioState:
    def __init__(self):
        self.history = []
        self.timestamps = []
    
    def add_state(self, state:Portfolio, timestamp):
        // TODO(): With some probability if current price is ITM, with some random prob we get option assigned 
        self.timestamps.append(timestamp)
        self.history.append(state)
