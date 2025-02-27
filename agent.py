from asset import Stock
from typing import List
from portfolio import Portfolio
from policy_interface import Policy

class Agent:
    def __init__(self, cash_amount, stock_tickers:List[Stock], stock_holdings:List[int], policies:List[Policy]):
        self.cash = cash_amount
        self._initialize_portfolio(cash_amount, stock_tickers, stock_holdings)
        self.poicies = policies
    
    def _initialize_portfolio(self, cash_amount, stock_tickers, stock_holdings):
        self.portfolio = Portfolio(cash_amount)
        for stock, quantity in zip(stock_tickers, stock_holdings):
            self.portfolio.buy_stock(stock, quantity)
    
    def step(self):
        raise NotImplementedError