# portfolio_class.py

from datetime import datetime
import pandas as pd
import numpy as np
from price_simulator import PriceSimulator

class Portfolio:
    def __init__(self, initial_cash=100.0):
        self.cash = initial_cash
        self.holdings_history = {}
        # Stock is represented as "STOCK", options as "CALL_STRIKE_MATURITY"
        self.current_holdings = {"STOCK": 0}
        self.transaction_history = []
        self.portfolio_value_history = {}
    
    def get_portfolio_value(self, price_simulator, current_date):
        """
        Calculate the total portfolio value at a given date.
        """
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        stock_value = self.current_holdings.get("STOCK", 0) * stock_price
        
        # Only include CALL options (their quantity will be negative for sold calls)
        options_value = 0
        for asset_id, quantity in self.current_holdings.items():
            if asset_id != "STOCK":
                option_type, strike, maturity = self._parse_option_id(asset_id)
                if option_type != "CALL":
                    continue
                maturity_date = datetime.strptime(maturity, "%Y-%m-%d")
                days_to_maturity = (maturity_date - current_date).days
                if days_to_maturity <= 0:
                    continue
                strike_val = float(strike)
                option_price = price_simulator.black_scholes_call(
                    strike_val, 
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

    def sell_covered_call(self, price_simulator, current_date, strike, maturity_date, quantity, contract_multiplier=1):

        required_shares = quantity * contract_multiplier
        available_shares = self.current_holdings.get("STOCK", 0)
        if available_shares < required_shares:
            # print("Not enough shares to cover the call sale.")
            return False
        
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        days_to_maturity = (maturity_date - current_date).days
        if days_to_maturity <= 0:
            return False
            
        option_price = price_simulator.black_scholes_call(
            strike, 
            days_to_maturity, 
            current_price=stock_price, 
            current_date=current_date
        )
        proceeds = quantity * option_price * contract_multiplier
        self.cash += proceeds
        
        # Record the short (sold) call position as negative quantity
        option_id = f"CALL_{strike}_{maturity_date.strftime('%Y-%m-%d')}"
        self.current_holdings[option_id] = self.current_holdings.get(option_id, 0) - quantity
        
        self.transaction_history.append({
            "Date": current_date,
            "Type": "SELL_COVERED_CALL",
            "Asset": option_id,
            "Quantity": quantity,
            "Price": option_price,
            "Total": proceeds,
            "Covered": True
        })
        self.holdings_history[current_date] = self.current_holdings.copy()
        # print(f"Sold {quantity} covered call contract(s) for {option_id}, covering {required_shares} shares.")
        return True

    def buy_stock(self, price_simulator, current_date, quantity):
        """
        Buy a specified quantity of the stock.
        """
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
            "Type": "BUY_STOCK",
            "Asset": "STOCK",
            "Quantity": quantity,
            "Price": stock_price,
            "Total": cost
        })
        self.holdings_history[current_date] = self.current_holdings.copy()
        return True

    def sell_stock(self, price_simulator, current_date, quantity):
        """
        Sell a specified quantity of the stock.
        """
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
            "Type": "SELL_STOCK",
            "Asset": "STOCK",
            "Quantity": quantity,
            "Price": stock_price,
            "Total": proceeds
        })
        self.holdings_history[current_date] = self.current_holdings.copy()
        return True

    def buy_option(self, price_simulator, current_date, option_type, strike, maturity_date, quantity):
        """
        Buy a call option (used here to close a sold call).
        """
        # print(f"buying to close {current_date} {option_type} {strike} {maturity_date} {quantity}")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if option_type != "CALL":
            raise ValueError("Only CALL options are allowed in this environment.")
        
        days_to_maturity = (maturity_date - current_date).days
        if days_to_maturity <= 0:
            raise ValueError("Option maturity must be in the future")
        
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        option_price = price_simulator.black_scholes_call(
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
        # When closing a sold call, buying reduces the negative position
        self.current_holdings[option_id] = self.current_holdings.get(option_id, 0) + quantity
        self.transaction_history.append({
            "Date": current_date,
            "Type": "CLOSE_COVERED_CALL",
            "Asset": option_id,
            "Quantity": quantity,
            "Price": option_price,
            "Total": cost
        })
        self.holdings_history[current_date] = self.current_holdings.copy()
        return True

    def get_portfolio_summary(self, price_simulator, current_date):
        """
        Get a summary of the current portfolio.
        """
        total_value = self.get_portfolio_value(price_simulator, current_date)
        value_details = self.portfolio_value_history[current_date]
        holdings = []
        stock_quantity = self.current_holdings.get("STOCK", 0)
        call_option_quantity = 0
        
        # Add stock holding
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
                if option_type != "CALL":
                    continue
                try:
                    date_str = current_date.strftime("%Y-%m-%d")
                    date_index = price_simulator.dates.index(date_str)
                    stock_price = price_simulator.simulated_prices.iloc[date_index]
                except (ValueError, IndexError):
                    raise ValueError(f"Date {date_str} not found in price simulator")
                maturity_date = datetime.strptime(maturity, "%Y-%m-%d")
                days_to_maturity = (maturity_date - current_date).days
                if days_to_maturity <= 0:
                    continue
                strike_val = float(strike)
                option_price = price_simulator.black_scholes_call(
                    strike_val, 
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
                call_option_quantity += quantity
        
        return {
            "Date": current_date,
            "Cash": value_details["Cash"],
            "Stock Value": value_details["Stock Value"],
            "Options Value": value_details["Options Value"],
            "Total Value": value_details["Total Value"],
            "Stock Quantity": stock_quantity,
            "Call Options Quantity": call_option_quantity,
            "Holdings": holdings
        }
    
    def get_transaction_history_df(self):
        return pd.DataFrame(self.transaction_history)

    def close_covered_calls_expiring_tomorrow(self, price_simulator, current_date):
        for asset_id, quantity in list(self.current_holdings.items()):
            if asset_id == "STOCK":
                continue

            option_type, strike, maturity_str = self._parse_option_id(asset_id)
            if option_type != "CALL":
                continue  # Only process CALL options

            if quantity >= 0:
                continue

            maturity_date = datetime.strptime(maturity_str, "%Y-%m-%d")
            days_to_maturity = (maturity_date - current_date).days

            if 0 < days_to_maturity <= 2:  
                qty_to_close = abs(quantity)
                success = self.buy_option(
                    price_simulator,
                    current_date,
                    "CALL",
                    float(strike),
                    maturity_date,
                    qty_to_close
                )
                if success:
                    pass
                    # print(f"Closed {qty_to_close} covered call contract(s) for {asset_id} expiring in {days_to_maturity} days.")
                else:
                    pass
                    # print(f"Failed to close covered call for {asset_id}.")
            elif days_to_maturity <= 0:
                pass
                # print(f"Skipping option {asset_id} - already expired.")
        
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
