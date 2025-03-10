from asset import Stock
from typing import List
# from portfolio import Portfolio, PortfolioState
from portfolio_class import Portfolio
from policy_interface import Policy
from price_simulator import PriceSimulator
import numpy as np
from typing import Dict, Any
import datetime

ACTION_TO_FUNCTION = {
    0: Portfolio.buy_option,
    1: Portfolio.sell_option,
    2: Portfolio.buy_stock,
    3: Portfolio.sell_stock
}

class Agent:
    def __init__(self, cash_amount, stock_tickers:List[Stock], stock_holdings:List[int], policies:List[Policy], horizon:int, action_dim:int, state_dim:int):
        #TODO(Maharshi): Add support for multiple policies
        self.policy = policies[0]
        self.cash = cash_amount
        self.simulators = {} #ticker to simulator
        self._initialize_simulators(stock_tickers)
        self.portfolio = self._initialize_portfolio(cash_amount, stock_tickers, stock_holdings)
        # self.portfolio = [self._initialize_portfolio(cash_amount, stock_tickers, stock_holdings) for _ in range(len(policies))]
        # self.portfolios_history = [PortfolioState() for i in range(len(policies))]
        # self.policies = policies
        # self.rewards = [[] for _ in range(len(policies))]
        self.time_step = 0
        self.horizon = horizon
        self.stock_to_agent_state = {stock.symbol:AgentState() for stock in stock_tickers}
        self.today = datetime.datetime.now()
    
    def _initialize_simulators(self, stock_tickers:List[Stock]):
        for stock in stock_tickers:
            self.simulators[stock.symbol] = PriceSimulator(stock.price, stock.mu, stock.sigma, stock.risk_free_rate)
    
    def _initialize_portfolio(self, cash_amount, stock_tickers, stock_holdings):
        portfolio = Portfolio(cash_amount)

        for stock, quantity in zip(stock_tickers, stock_holdings):
            portfolio.buy_stock(self.simulators[stock.symbol],stock, quantity)
        
        return portfolio

    def step(self):
        #update states
        for stock in self.stock_to_agent_state.keys():
            state = self.stock_to_agent_state[stock]
            if len(self.rewards) == 0:
                self.stock_to_agent_state[stock].update_state(np.zeros(4), self.portfolio.get_state(self.simulators[stock], self.today+datetime.timedelta(days=self.time_step)), 0, self.time_step)
            actions = self.policies.get_action(state)

            state.update_state(np.self.portfolio.get_state(), self.portfolio.get_value(), self.time_step)

        for i in range(len(self.policies)):
            # buy_quantity, sell_quantity
            # 0: buy, 1: sell 
            #TODO(Maharshi): Add support for multiple stocks in the portfolio, currently assuming only one stock
            # for now only 30 day options
            buy_option_quantity,sell_option_quantity = actions[i]
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
    def __init__(self, stock:str, price_simulator:PriceSimulator):
        self.stock=stock
        self.price_simulator = price_simulator
        self.actions = []
        self.price_history = []
        self.portfolio_state = []
        self.rewards = []
    
    def update_state(self, action:np.ndarray, portolio_state:Dict[str,Any], reward:float, time_step:int):
        self.price_history.append(self.price_simulator.get_price_history(time_step))
        self.actions.append(action)
        self.portfolio_state.append(portolio_state)
        self.rewards.append(reward)