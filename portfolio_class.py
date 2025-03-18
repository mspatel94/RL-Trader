from datetime import datetime
import pandas as pd
import numpy as np
from price_simulator import PriceSimulator

class Portfolio:
    def __init__(self, initial_cash=100.0):
        """
        Initialize a portfolio.
        
        Parameters:
        initial_cash (float): Initial cash amount in the portfolio
        """
        self.cash = initial_cash
        # Dictionary to track holdings at each timestamp
        self.holdings_history = {}
        # Current holdings: {asset_id: quantity}
        # Stock is represented as "STOCK"
        # Options are represented as "CALL_STRIKE_MATURITY" or "PUT_STRIKE_MATURITY"
        self.current_holdings = {"STOCK": 0}
        self.transaction_history = []
        self.portfolio_value_history = {}
    
    def get_portfolio_value(self, price_simulator, current_date):
        """
        Calculate the total value of the portfolio at a given date.
        
        Parameters:
        price_simulator (PriceSimulator): The price simulator with current stock prices
        current_date (datetime): The date for which to calculate the portfolio value
        
        Returns:
        float: Total portfolio value in dollars
        """
        if price_simulator.simulated_prices is None:
            raise ValueError("Price simulator must have run simulate_path first")
        
        # Get the stock price for the current date
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        # Calculate stock value
        stock_value = self.current_holdings.get("STOCK", 0) * stock_price
        
        # Calculate options value
        options_value = 0
        for asset_id, quantity in self.current_holdings.items():
            if asset_id != "STOCK":
                option_type, strike, maturity = self._parse_option_id(asset_id)
                
                # Calculate days to maturity
                maturity_date = datetime.strptime(maturity, "%Y-%m-%d")
                days_to_maturity = (maturity_date - current_date).days
                
                # Skip expired options
                if days_to_maturity <= 0:
                    continue
                    
                # Calculate option price
                strike = float(strike)
                if option_type == "CALL":
                    option_price = price_simulator.black_scholes_call(
                        strike, 
                        days_to_maturity, 
                        current_price=stock_price, 
                        current_date=current_date
                    )
                else:  # PUT
                    option_price = price_simulator.black_scholes_put(
                        strike, 
                        days_to_maturity, 
                        current_price=stock_price, 
                        current_date=current_date
                    )
                
                options_value += quantity * option_price
        
        # Total portfolio value
        total_value = self.cash + stock_value + options_value
        
        # Store the portfolio value for this date
        self.portfolio_value_history[current_date] = {
            "Cash": self.cash,
            "Stock Value": stock_value,
            "Options Value": options_value,
            "Total Value": total_value,
            "Stock Price": stock_price
        }
        
        return total_value
    
    def _parse_option_id(self, option_id):
        """
        Parse an option ID into its components.
        
        Parameters:
        option_id (str): Option ID in format "CALL_STRIKE_MATURITY" or "PUT_STRIKE_MATURITY"
        
        Returns:
        tuple: (option_type, strike, maturity)
        """
        parts = option_id.split("_")
        option_type = parts[0]  # CALL or PUT
        strike = parts[1]       # Strike price
        maturity = parts[2]     # Maturity date in YYYY-MM-DD format
        
        return option_type, strike, maturity

    def sell_covered_call(self, price_simulator, current_date, strike, maturity_date, quantity, contract_multiplier=1):
        """
        Sell call options as covered calls. Checks that there are enough underlying shares.
        
        Parameters:
            price_simulator (PriceSimulator): To price the option.
            current_date (datetime): Date of the transaction.
            strike (float): Strike price for the call option.
            maturity_date (datetime): Expiration date of the option.
            quantity (int): Number of call contracts to sell.
            contract_multiplier (int): Number of shares per contract (default 1).
        
        Returns:
            bool: True if the transaction is successful, False otherwise.
        """
        # Check if enough stock is available to cover the call(s)
        required_shares = quantity * contract_multiplier
        available_shares = self.current_holdings.get("STOCK", 0)
        if available_shares < required_shares:
            print("Not enough shares to cover the call sale.")
            return False
        
        # Get the current stock price
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        # Calculate days to maturity
        days_to_maturity = (maturity_date - current_date).days
        # Price the call option
        option_price = price_simulator.black_scholes_call(
            strike, 
            days_to_maturity, 
            current_price=stock_price, 
            current_date=current_date
        )
        # Calculate proceeds (using contract_multiplier)
        proceeds = quantity * option_price * contract_multiplier
        
        # Update cash with the premium received
        self.cash += proceeds
        
        # Record the covered call sale.
        # Here we record a short position in the option as a negative quantity.
        option_id = f"CALL_{strike}_{maturity_date.strftime('%Y-%m-%d')}"
        self.current_holdings[option_id] = self.current_holdings.get(option_id, 0) - quantity
        
        # Optionally, you may want to mark the underlying stock as "covered"
        # so that they are not sold again until the option expires.
        # For simplicity, we leave the stock position unchanged.
        
        # Record the transaction
        self.transaction_history.append({
            "Date": current_date,
            "Type": "SELL_COVERED_CALL",
            "Asset": option_id,
            "Quantity": quantity,
            "Price": option_price,
            "Total": proceeds,
            "Covered": True
        })
        
        print(f"Sold {quantity} covered call contract(s) for {option_id}, covering {required_shares} shares.")
        return True


    def sell_options_expiring_tomorrow(self, price_simulator, current_date):
        """
        Sell all option contracts in the portfolio that are expiring in one day.
        
        Parameters:
            price_simulator (PriceSimulator): The simulator to obtain current stock prices and option prices.
            current_date (datetime): The current date at which to check for expiring options.
            
        Returns:
            None
        """
        # Iterate over a copy of the holdings keys since we might modify the dict.
        for asset_id, quantity in list(self.current_holdings.items()):
            # Skip stock holdings
            if asset_id == "STOCK":
                continue

            # Parse the option id to get option type, strike, and maturity date.
            option_type, strike, maturity_str = self._parse_option_id(asset_id)
            maturity_date = datetime.strptime(maturity_str, "%Y-%m-%d")
            
            # Calculate days until maturity
            days_to_maturity = (maturity_date - current_date).days
            
            # If the option is expiring in one day, sell all contracts.
            if days_to_maturity <= 2:
                # Attempt to sell all held contracts for this option.
                success = self.sell_option(
                    price_simulator, 
                    current_date, 
                    option_type, 
                    float(strike), 
                    maturity_date, 
                    quantity
                )
                if success:
                    print(f"Sold {quantity} contract(s) of {asset_id} expiring in 1 day.")
                else:
                    print(f"Failed to sell {asset_id}.")
    
    def buy_stock(self, price_simulator, current_date, quantity):
        """
        Buy a specified quantity of the stock.
        
        Parameters:
        price_simulator (PriceSimulator): The price simulator with current stock prices
        current_date (datetime): The date of the transaction
        quantity (int): Number of shares to buy
        
        Returns:
        bool: True if the transaction was successful, False otherwise
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        # Get the stock price for the current date
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        # Calculate the cost
        cost = quantity * stock_price
        
        # Check if there's enough cash
        if cost > self.cash:
            return False
        
        # Update cash
        self.cash -= cost
        
        # Update holdings
        self.current_holdings["STOCK"] = self.current_holdings.get("STOCK", 0) + quantity
        
        # Record the transaction
        self.transaction_history.append({
            "Date": current_date,
            "Type": "BUY",
            "Asset": "STOCK",
            "Quantity": quantity,
            "Price": stock_price,
            "Total": cost
        })
        
        # Update holdings history for this date
        self.holdings_history[current_date] = self.current_holdings.copy()
        
        return True
    
    def sell_stock(self, price_simulator, current_date, quantity):
        """
        Sell a specified quantity of the stock.
        
        Parameters:
        price_simulator (PriceSimulator): The price simulator with current stock prices
        current_date (datetime): The date of the transaction
        quantity (int): Number of shares to sell
        
        Returns:
        bool: True if the transaction was successful, False otherwise
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        # Check if there are enough shares
        current_quantity = self.current_holdings.get("STOCK", 0)
        if quantity > current_quantity:
            return False
        
        # Get the stock price for the current date
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        # Calculate the proceeds
        proceeds = quantity * stock_price
        
        # Update cash
        self.cash += proceeds
        
        # Update holdings
        self.current_holdings["STOCK"] = current_quantity - quantity
        
        # Record the transaction
        self.transaction_history.append({
            "Date": current_date,
            "Type": "SELL",
            "Asset": "STOCK",
            "Quantity": quantity,
            "Price": stock_price,
            "Total": proceeds
        })
        
        # Update holdings history for this date
        self.holdings_history[current_date] = self.current_holdings.copy()
        
        return True
    
    def buy_option(self, price_simulator, current_date, option_type, strike, maturity_date, quantity):
        """
        Buy a specified quantity of options.
        
        Parameters:
        price_simulator (PriceSimulator): The price simulator with current stock prices
        current_date (datetime): The date of the transaction
        option_type (str): "CALL" or "PUT"
        strike (float): Strike price
        maturity_date (datetime): Maturity date of the option
        quantity (int): Number of option contracts to buy
        
        Returns:
        bool: True if the transaction was successful, False otherwise
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        if option_type not in ["CALL", "PUT"]:
            raise ValueError("Option type must be either 'CALL' or 'PUT'")
        
        # Calculate days to maturity
        days_to_maturity = (maturity_date - current_date).days
        if days_to_maturity <= 0:
            raise ValueError("Option maturity must be in the future")
        
        # Get the current stock price
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        print(date_str, current_date, days_to_maturity, stock_price)
        # Calculate option price
        if option_type == "CALL":
            option_price = price_simulator.black_scholes_call(
                strike, 
                days_to_maturity, 
                current_price=stock_price, 
                current_date=current_date
            )
        else:  # PUT
            option_price = price_simulator.black_scholes_put(
                strike, 
                days_to_maturity, 
                current_price=stock_price, 
                current_date=current_date
            )
        
        # Calculate the cost (assuming 1 contract = 1 shares)
        cost = quantity * option_price
        
        # Check if there's enough cash
        if cost > self.cash:
            return False
        
        # Create option ID
        maturity_str = maturity_date.strftime("%Y-%m-%d")
        option_id = f"{option_type}_{strike}_{maturity_str}"
        
        # Update cash
        self.cash -= cost
        
        # Update holdings
        self.current_holdings[option_id] = self.current_holdings.get(option_id, 0) + quantity
        
        # Record the transaction
        self.transaction_history.append({
            "Date": current_date,
            "Type": "BUY",
            "Asset": option_id,
            "Quantity": quantity,
            "Price": option_price,
            "Total": cost
        })
        
        # Update holdings history for this date
        self.holdings_history[current_date] = self.current_holdings.copy()
        
        return True
    
    def sell_option(self, price_simulator, current_date, option_type, strike, maturity_date, quantity):
        """
        Sell a specified quantity of options.
        
        Parameters:
        price_simulator (PriceSimulator): The price simulator with current stock prices
        current_date (datetime): The date of the transaction
        option_type (str): "CALL" or "PUT"
        strike (float): Strike price
        maturity_date (datetime): Maturity date of the option
        quantity (int): Number of option contracts to sell
        
        Returns:
        bool: True if the transaction was successful, False otherwise
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        if option_type not in ["CALL", "PUT"]:
            raise ValueError("Option type must be either 'CALL' or 'PUT'")
        
        # Create option ID
        maturity_str = maturity_date.strftime("%Y-%m-%d")
        option_id = f"{option_type}_{strike}_{maturity_str}"
        
        # Check if there are enough contracts
        current_quantity = self.current_holdings.get(option_id, 0)
        if quantity > current_quantity:
            return False
        
        # Calculate days to maturity
        days_to_maturity = (maturity_date - current_date).days
        if days_to_maturity <= 0:
            raise ValueError("Cannot sell expired options")
        
        # Get the current stock price
        try:
            date_str = current_date.strftime("%Y-%m-%d")
            date_index = price_simulator.dates.index(date_str)
            stock_price = price_simulator.simulated_prices.iloc[date_index]
        except (ValueError, IndexError):
            raise ValueError(f"Date {date_str} not found in price simulator")
        
        # Calculate option price
        if option_type == "CALL":
            option_price = price_simulator.black_scholes_call(
                strike, 
                days_to_maturity, 
                current_price=stock_price, 
                current_date=current_date
            )
        else:  # PUT
            option_price = price_simulator.black_scholes_put(
                strike, 
                days_to_maturity, 
                current_price=stock_price, 
                current_date=current_date
            )
        
        # Calculate the proceeds (assuming 1 contract = 1 shares)
        proceeds = quantity * option_price
        
        # Update cash
        self.cash += proceeds
        
        # Update holdings
        self.current_holdings[option_id] = current_quantity - quantity
        
        # Remove if quantity is 0
        if self.current_holdings[option_id] == 0:
            del self.current_holdings[option_id]
        
        # Record the transaction
        self.transaction_history.append({
            "Date": current_date,
            "Type": "SELL",
            "Asset": option_id,
            "Quantity": quantity,
            "Price": option_price,
            "Total": proceeds
        })
        
        # Update holdings history for this date
        self.holdings_history[current_date] = self.current_holdings.copy()
        
        return True
    
    def get_portfolio_summary(self, price_simulator, current_date):
        """
        Get a summary of the current portfolio.
        
        Parameters:
        price_simulator (PriceSimulator): The price simulator with current stock prices
        current_date (datetime): The date for which to generate the summary
        
        Returns:
        dict: Portfolio summary information
        """
        # Calculate total value first (will update portfolio_value_history)
        total_value = self.get_portfolio_value(price_simulator, current_date)
        
        # Get the detailed breakdown from the history
        value_details = self.portfolio_value_history[current_date]

        
        # Get holdings details
        holdings = []
        
        # Add stock holding
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
        
        # Add options holdings
        for asset_id, quantity in self.current_holdings.items():
            if asset_id != "STOCK":
                option_type, strike, maturity = self._parse_option_id(asset_id)
                
                # Calculate days to maturity
                maturity_date = datetime.strptime(maturity, "%Y-%m-%d")
                days_to_maturity = (maturity_date - current_date).days
                
                if option_type == "CALL":
                    call_option_quantity += quantity
                else:
                    put_option_quantity += quantity

                # Skip expired options
                if days_to_maturity <= 0:
                    continue
                    
                # Get the current stock price
                try:
                    date_str = current_date.strftime("%Y-%m-%d")
                    date_index = price_simulator.dates.index(date_str)
                    stock_price = price_simulator.simulated_prices.iloc[date_index]
                except (ValueError, IndexError):
                    raise ValueError(f"Date {date_str} not found in price simulator")
                    
                # Calculate option price
                strike = float(strike)
                if option_type == "CALL":
                    option_price = price_simulator.black_scholes_call(
                        strike, 
                        days_to_maturity, 
                        current_price=stock_price, 
                        current_date=current_date
                    )
                else:  # PUT
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
        """
        Get the history of the value of a specific asset in the portfolio.
        
        Parameters:
        asset_id (str): The asset ID to get the value history for
        
        Returns:
        dict: A dictionary of date to value for the asset
        """

        df = self.get_portfolio_value_history_df()
        return df[df['Asset'] == asset_id]
    
    def get_transaction_history_df(self):
        """
        Get the transaction history as a pandas DataFrame.
        
        Returns:
        pandas.DataFrame: Transaction history
        """
        return pd.DataFrame(self.transaction_history)
    
    def get_portfolio_value_history_df(self):
        """
        Get the portfolio value history as a pandas DataFrame.
        
        Returns:
        pandas.DataFrame: Portfolio value history
        """
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