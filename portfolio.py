#TODO: Add random variable that makes us sell the stock at each time step

class Portfolio:
    def __init__(self, cash_amount):
        self.cash = cash_amount
        self.stocks = [] # Stock Object
        self.options = [] # Options Object

    
    
    def buy_stock(self, stock, quantity):
        //TODO(Farzad): start tracking prices in a data structure by calling price simulator
        raise NotImplementedError
    
    def sell_stock(self, stock, quantity):
        raise NotImplementedError

    # portfolio_option_count = 0 -> buy -> buy_open -> count > 0
    # portfolio_option_count > 0 -> buy -> buy_close -> count = 0
    # portfolio_option_count = 0 -> sell -> sell_open -> count < 0
    # portfolio_option_count < 0 -> sell -> sell_close -> count = 0
    
    def buy_to_close_option(self, option, quantity):
        raise NotImplementedError

    #Buy
    def sell_option_to_open(self, option, quantity):
        raise NotImplementedError

    

class PortfolioState:
    def __init__(self):
        self.history = []
        self.timestamps = []
    
    def add_state(self, state:Portfolio, timestamp):
        # TODO(): With some probability if current price is ITM, with some random prob we get option assigned 
        # WE ASSUME EUROPEAN STYLE 
        self.timestamps.append(timestamp)
        self.history.append(state)
