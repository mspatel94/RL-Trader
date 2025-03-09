from asset import Stock
from typing import List
from portfolio import Portfolio, PortfolioState
from policy_interface import Policy


ACTION_TO_FUNCTION = {
    0: Portfolio.buy_option,
    1: Portfolio.sell_option,
}

class Agent:
    def __init__(self, cash_amount, stock_tickers:List[Stock], stock_holdings:List[int], policies:List[Policy], horizon:int):
        self.cash = cash_amount
        self.portfolios = [self._initialize_portfolio(cash_amount, stock_tickers, stock_holdings) for _ in range(len(policies))]
        self.portfolios_history = [PortfolioState() for i in range(len(policies))]
        self.policies = policies
        self.rewards = [[] for _ in range(len(policies))]
        self.time_step = 0
        self.horizon = horizon
    
    def _initialize_portfolio(self, cash_amount, stock_tickers, stock_holdings):
        portfolio = Portfolio(cash_amount)
        for stock, quantity in zip(stock_tickers, stock_holdings):
            portfolio.buy_stock(stock, quantity)
        
        return portfolio

    def step(self):
        #update states
        for i in range(len(self.policies)):
            self.portfolios_history[i].add_state(self.portfolios[i], self.time_step)
        actions = [self.policies[i].get_action(self.portfolios_history[i].get_state()) for i in range(len(self.policies))]

        for i in range(len(self.policies)):
            # buy_quantity, sell_quantity
            # 0: buy, 1: sell 
            #TODO(Maharshi): Add support for multiple stocks in the portfolio, currently assuming only one stock
            buy_quantity,sell_quantity = actions[i]
            true_reward = self.portfolios[i].get_value()
            self.portfolios[i].sell_option('dummy_place_holder', sell_quantity)
            self.portfolios[i].buy_option('dummy_place_holder', buy_quantity)
            self.portfolios[i].step()
            true_reward = self.portfolios[i].get_value() - true_reward
            self.rewards[i].append(true_reward)
            
        self.time_step += 1
    
    def run(self):
        while self.time_step < self.horizon:
            self.step()

class AgentState:
    def __init__(self):
        self.actions = []
        self.history = []
        self.rewards = []
    
    def update_state(self, action, portolio:Portfolio, reward):
        self.actions.append(action)
        self.history.append(portolio)
        self.rewards.append(reward)