from datetime import datetime
import pandas as pd
import numpy as np
from price_simulator import PriceSimulator

class Portfolio:
    def __init__(self, initial_cash=100.0):
        self.cash = initial_cash
        self.holdings_history = {}
        self.current_holdings = {"STOCK": 0}
        self.transaction_history = []
        self.portfolio_value_history = {}
    
    def get_portfolio_value(self, price_simulator, current_date):
        if price_simulator.simulated_prices is None:
            raise ValueError("Price simulator must have run simulate_path first")
        
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        stock_value = self.current_holdings.get("STOCK", 0) * stock_price
        
        options_value = 0
        for asset_id, quantity in self.current_holdings.items():
            if asset_id != "STOCK":
                option_type, strike, maturity = self._parse_option_id(asset_id)
                
                maturity_date = datetime.strptime(maturity, "%Y-%m-%d")
                days_to_maturity = (maturity_date - current_date).days
                
                if days_to_maturity <= 0:
                    continue
                    
                strike = float(strike)
                if option_type == "CALL":
                    option_price = price_simulator.black_scholes_call(
                        strike, 
                        days_to_maturity, 
                        current_price=stock_price, 
                        current_date=current_date
                    )
                else:  
                    option_price = price_simulator.black_scholes_put(
                        strike, 
                        days_to_maturity, 
                        current_price=stock_price, 
                        current_date=current_date
                    )
                
                options_value += quantity * option_price
        
        total_value = self.cash + stock_value + options_value
        
        self.portfolio_value_history[current_date] = {
            "Cash": self.cash,
            "Stock Value": stock_value,
            "Options Value": options_value,
            "Total Value": total_value,
            "Stock Price": stock_price
        }
        
        return total_value
    
    def _parse_option_id(self, option_id):
        parts = option_id.split("_")
        option_type = parts[0]  
        strike = parts[1]       
        maturity = parts[2]
        
        return option_type, strike, maturity
    
    def buy_stock(self, price_simulator, current_date, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        cost = quantity * stock_price
        if cost > self.cash:
            return False
        
        self.cash -= cost
        
        self.current_holdings["STOCK"] = self.current_holdings.get("STOCK", 0) + quantity
        
        self.transaction_history.append({
            "Date": current_date,
            "Type": "BUY",
            "Asset": "STOCK",
            "Quantity": quantity,
            "Price": stock_price,
            "Total": cost
        })
        
        self.holdings_history[current_date] = self.current_holdings.copy()
        
        return True
    
    def sell_stock(self, price_simulator, current_date, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        current_quantity = self.current_holdings.get("STOCK", 0)
        if quantity > current_quantity:
            return False
        
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        proceeds = quantity * stock_price
        self.cash += proceeds
        self.current_holdings["STOCK"] = current_quantity - quantity
        
        self.transaction_history.append({
            "Date": current_date,
            "Type": "SELL",
            "Asset": "STOCK",
            "Quantity": quantity,
            "Price": stock_price,
            "Total": proceeds
        })
        
        self.holdings_history[current_date] = self.current_holdings.copy()
        
        return True
    
    def buy_option(self, price_simulator, current_date, option_type, strike, maturity_date, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        if option_type not in ["CALL", "PUT"]:
            raise ValueError("Option type must be either 'CALL' or 'PUT'")
        
        days_to_maturity = (maturity_date - current_date).days
        if days_to_maturity <= 0:
            raise ValueError("Option maturity must be in the future")
        
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        print(date_str, current_date, days_to_maturity, stock_price)
        if option_type == "CALL":
            option_price = price_simulator.black_scholes_call(
                strike, 
                days_to_maturity, 
                current_price=stock_price, 
                current_date=current_date
            )
        else:  
            option_price = price_simulator.black_scholes_put(
                strike, 
                days_to_maturity, 
                current_price=stock_price, 
                current_date=current_date
            )
        
        cost = quantity * option_price
        
        if cost > self.cash:
            return False
        
        maturity_str = maturity_date.strftime("%Y-%m-%d")
        option_id = f"{option_type}_{strike}_{maturity_str}"
        self.cash -= cost
        self.current_holdings[option_id] = self.current_holdings.get(option_id, 0) + quantity
        
        self.transaction_history.append({
            "Date": current_date,
            "Type": "BUY",
            "Asset": option_id,
            "Quantity": quantity,
            "Price": option_price,
            "Total": cost
        })
        
        self.holdings_history[current_date] = self.current_holdings.copy()
        
        return True
    
    def sell_option(self, price_simulator, current_date, option_type, strike, maturity_date, quantity):
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        if option_type not in ["CALL", "PUT"]:
            raise ValueError("Option type must be either 'CALL' or 'PUT'")
        
        maturity_str = maturity_date.strftime("%Y-%m-%d")
        option_id = f"{option_type}_{strike}_{maturity_str}"
        
        current_quantity = self.current_holdings.get(option_id, 0)
        if quantity > current_quantity:
            return False
        
        days_to_maturity = (maturity_date - current_date).days
        if days_to_maturity <= 0:
            raise ValueError("Cannot sell expired options")
        
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        if option_type == "CALL":
            option_price = price_simulator.black_scholes_call(
                strike, 
                days_to_maturity, 
                current_price=stock_price, 
                current_date=current_date
            )
        else:  
            option_price = price_simulator.black_scholes_put(
                strike, 
                days_to_maturity, 
                current_price=stock_price, 
                current_date=current_date
            )
        
        proceeds = quantity * option_price
        self.cash += proceeds
        self.current_holdings[option_id] = current_quantity - quantity
        if self.current_holdings[option_id] == 0:
            del self.current_holdings[option_id]
        
        self.transaction_history.append({
            "Date": current_date,
            "Type": "SELL",
            "Asset": option_id,
            "Quantity": quantity,
            "Price": option_price,
            "Total": proceeds
        })
        
        self.holdings_history[current_date] = self.current_holdings.copy()
        
        return True
    
    def get_portfolio_summary(self, price_simulator, current_date):
        total_value = self.get_portfolio_value(price_simulator, current_date)
        value_details = self.portfolio_value_history[current_date]

        holdings = []
        
        stock_quantity = self.current_holdings.get("STOCK", 0)
        call_option_quantity = 0
        put_option_quantity = 0
        if stock_quantity > 0:
            try:
                date_str = current_date.strftime("%Y-%m-%d")
                date_index = price_simulator.dates.index(date_str)
                stock_price = price_simulator.simulated_prices.iloc[date_index]
                holdings.append({
                    "Asset": "STOCK",
                    "Quantity": stock_quantity,
                    "Price": stock_price,
                    "Value": stock_quantity * stock_price
                })
            except (ValueError, IndexError):
                pass
        
        for asset_id, quantity in self.current_holdings.items():
            if asset_id != "STOCK":
                option_type, strike, maturity = self._parse_option_id(asset_id)
                
                maturity_date = datetime.strptime(maturity, "%Y-%m-%d")
                days_to_maturity = (maturity_date - current_date).days
                
                if option_type == "CALL":
                    call_option_quantity += quantity
                else:
                    put_option_quantity += quantity

                if days_to_maturity <= 0:
                    continue
                    
                try:
                    date_str = current_date.strftime("%Y-%m-%d")
                    date_index = price_simulator.dates.index(date_str)
                    stock_price = price_simulator.simulated_prices.iloc[date_index]
                except (ValueError, IndexError):
                    raise ValueError(f"Date {date_str} not found in price simulator")
                    
                strike = float(strike)
                if option_type == "CALL":
                    option_price = price_simulator.black_scholes_call(
                        strike, 
                        days_to_maturity, 
                        current_price=stock_price, 
                        current_date=current_date
                    )
                else:  
                    option_price = price_simulator.black_scholes_put(
                        strike, 
                        days_to_maturity, 
                        current_price=stock_price, 
                        current_date=current_date
                    )
                
                holdings.append({
                    "Asset": asset_id,
                    "Quantity": quantity,
                    "Price": option_price,
                    "Value": quantity * option_price
                })
        
        return {
            "Date": current_date,
            "Cash": value_details["Cash"],
            "Stock Value": value_details["Stock Value"],
            "Options Value": value_details["Options Value"],
            "Total Value": value_details["Total Value"],
            "Stock Quantity": stock_quantity,
            "Call Options Quantity": call_option_quantity,
            "Put Options Quantity": put_option_quantity,
            "Holdings": holdings
        }
    
    def get_value_history(self, asset_id:str):
        df = self.get_portfolio_value_history_df()
        return df[df['Asset'] == asset_id]
    
    def get_transaction_history_df(self):
        return pd.DataFrame(self.transaction_history)
    
    def get_portfolio_value_history_df(self):
        history_list = []
        for date, values in self.portfolio_value_history.items():
            history_list.append({
                "Date": date,
                "Cash": values["Cash"],
                "Stock Value": values["Stock Value"],
                "Options Value": values["Options Value"],
                "Total Value": values["Total Value"],
                "Stock Price": values["Stock Price"]
            })
        
        return pd.DataFrame(history_list)