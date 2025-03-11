from asset import Stock
from typing import List
# from portfolio import Portfolio, PortfolioState
from portfolio_class import Portfolio
from policy_interface import Policy
from price_simulator import PriceSimulator
import numpy as np
from typing import Dict, Any
import datetime

VALID_ACTIONS_MAP = {
    0: 'buy',
    1: 'sell',
    2: 'hold'
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
        self.stock_to_agent_state = {stock.symbol:AgentState(self.simulators[stock],self.portfolio) for stock in stock_tickers}
        self.today = datetime.datetime.now()
    
    def _initialize_simulators(self, stock_tickers:List[Stock]):
        for stock in stock_tickers:
            self.simulators[stock.symbol] = PriceSimulator(stock.price, stock.mu, stock.sigma, stock.risk_free_rate)
    
    def _initialize_portfolio(self, cash_amount, stock_tickers, stock_holdings):
        portfolio = Portfolio(cash_amount)

        for stock, quantity in zip(stock_tickers, stock_holdings):
            portfolio.buy_stock(self.simulators[stock.symbol],stock, quantity)
        
        return portfolio

    def _get_current_date(self):
        return self.today+datetime.timedelta(days=self.time_step)

    def _execute_action(self, action):
        if round(action[0]) not in VALID_ACTIONS_MAP:
            raise ValueError(f"Invalid action: {action}")
        
        action_type = VALID_ACTIONS_MAP[round(action[0])]
        action_amount = action[1]
        if action_type == 'buy':
            self.portfolio.buy_stock(self.simulators["STOCK"], self._get_current_date(), action_amount)
        elif action_type == 'sell':
            self.portfolio.sell_stock(self.simulators["STOCK"], self._get_current_date(), action_amount)
        else:
            print("Holding")
        
    def step(self):
        #update states
        for stock in self.stock_to_agent_state.keys():
            state = self.stock_to_agent_state[stock]
            if len(self.rewards) == 0:
                state.update_state(np.zeros(4), self.portfolio.get_state(self.simulators[stock], self.today+datetime.timedelta(days=self.time_step)), 0, self.time_step)
            true_reward = self.portfolios[i].get_value()

            action = self.policy.get_action(state.get_stock_state())
            self._execute_action(action)
            true_reward = self.portfolios[i].get_value() - true_reward

            state.update_state(action, true_reward, self.time_step)
        self.time_step+=1
    
    def run(self):
        while self.time_step < self.horizon:
            self.step()


class StockState:
    def __init__(self, price_history:List[float], cash:float, stocks_owned:float, stocks_value:float, portfolio_value:float):
        self.price_history = price_history
        self.cash = cash
        self.stocks_owned = stocks_owned
        self.stocks_value = stocks_value
        self.portfolio_value = portfolio_value

class OptionsState:
    def __init__(self, options_owned:float, options_value:float, portfolio_value:float, stock_price_history:List[float], option_price_history:List[float], time_to_maturity:int):
        self.options_owned = options_owned
        self.options_value = options_value
        self.portfolio_value = portfolio_value
        self.stock_price_history = stock_price_history
        self.option_price_history = option_price_history
        self.time_to_maturity = time_to_maturity
class AgentState:
    def __init__(self, stock:str, price_simulator:PriceSimulator, portfolio:Portfolio):
        self.stock=stock
        self.price_simulator = price_simulator
        self.actions = []
        self.portfolio = portfolio
        self.portfolio_state = []
        self.rewards = []
    
    def update_state(self, action, reward:float, time_step:int):
        self.price_history.append(self.price_simulator.get_price_history(time_step))
        self.actions.append(action)
        self.rewards.append(reward)
    
    def get_stock_state(self, date):
        price_history = self.portfolio.get_portfolio_value_history_df()
        price_history_list = price_history[price_history['date'] <= date]['Stock Price'].tolist()
        stocks_owned = self.portfolio.get_portfolio_summary(date)['Stock Quantity']
        stocks_value = self.portfolio.get_portfolio_summary(date)['Stock Value']
        cash = self.portfolio.get_portfolio_summary(date)['Cash']
        total_value = self.portfolio.get_portfolio_summary(date)['Total Value']
        return StockState(price_history_list, cash, stocks_owned, stocks_value, total_value)

if __name__=="__main__":
    stock_ticker = 'AAPL'
    starting_price = 100
    mu = 0.1
    sigma = 0.2
    risk_free_rate = 0.01
    horizon = 1000
    state_dim = 10
    action_dim = 2
    
    cash = 1000
    stock = Stock(stock_ticker, starting_price, mu, sigma, risk_free_rate)
    agent = Agent(cash, [stock], [1],[None], horizon, action_dim, state_dim)
    