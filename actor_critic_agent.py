import numpy as np
import torch
from actor_critic_model import ActorCritic
from agent import Agent, Stock
import matplotlib.pyplot as plt
import datetime

def run_actor_critic_agent():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize parameters
    stock_ticker = 'AAPL'
    starting_price = 100
    mu = 0.1
    sigma = 0.2
    risk_free_rate = 0.01
    horizon = 252  # One trading year
    
    # State dimension: price history + portfolio features
    history_length = 30
    state_dim = history_length + 4  # price history + cash, stocks_owned, stocks_value, portfolio_value
    
    # Action dimension: [action_type, amount]
    action_dim = 2
    
    # Initialize cash and stock
    cash = 10000
    stock = Stock(stock_ticker, starting_price, mu, sigma, risk_free_rate)
    
    # Initialize policy (Actor-Critic)
    policy = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        exploration_noise=0.5,
        min_exploration=0.05,
        exploration_decay=0.995
    )
    
    # Initialize agent
    agent = Agent(cash, [stock], [0], [policy], horizon, action_dim, state_dim)
    
    # Run simulation
    print("Starting Actor-Critic agent simulation...")
    portfolio_values = []
    cash_values = []
    stock_quantities = []
    stock_prices = []
    action_types = []
    action_amounts = []
    
    for step in range(horizon):
        # Step the agent
        agent.step()
        
        # Record portfolio state
        current_date = agent._get_current_date()
        portfolio_summary = agent.portfolio.get_portfolio_summary(
            agent.simulators[stock_ticker], 
            current_date.strftime("%Y-%m-%d")
        )
        
        portfolio_values.append(portfolio_summary['Total Value'])
        cash_values.append(portfolio_summary['Cash'])
        stock_quantities.append(portfolio_summary['Stock Quantity'])
        
        # Get stock price
        stock_price = agent.simulators[stock_ticker].get_price_at_date(current_date.strftime("%Y-%m-%d"))
        stock_prices.append(stock_price)
        
        # Record action
        if step < len(policy.actions):
            action = policy.actions[step]
            action_types.append(int(action[0]))
            action_amounts.append(float(action[1]))
        
        # Print progress
        if step % 10 == 0 or step == horizon - 1:
            print(f"Step {step+1}/{horizon}")
            print(f"Date: {current_date.strftime('%Y-%m-%d')}")
            print(f"Stock Price: ${stock_price:.2f}")
            print(f"Cash: ${portfolio_summary['Cash']:.2f}")
            print(f"Stock Value: ${portfolio_summary['Stock Value']:.2f}")
            print(f"Stock Quantity: {portfolio_summary['Stock Quantity']}")
            print(f"Total Portfolio Value: ${portfolio_summary['Total Value']:.2f}")
            if step > 0:
                print(f"Return since start: {portfolio_summary['Total Value']/portfolio_values[0] - 1:.2%}")
            print("=" * 50)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot portfolio value
    plt.subplot(2, 2, 1)
    plt.plot(portfolio_values)
    plt.title('Portfolio Value')
    plt.xlabel('Time Step')
    plt.ylabel('Value ($)')
    
    # Plot cash
    plt.subplot(2, 2, 2)
    plt.plot(cash_values)
    plt.title('Cash')
    plt.xlabel('Time Step')
    plt.ylabel('Value ($)')
    
    # Plot stock quantity
    plt.subplot(2, 2, 3)
    plt.plot(stock_quantities)
    plt.title('Stock Quantity')
    plt.xlabel('Time Step')
    plt.ylabel('Quantity')
    
    # Plot stock price
    plt.subplot(2, 2, 4)
    plt.plot(stock_prices)
    plt.title('Stock Price')
    plt.xlabel('Time Step')
    plt.ylabel('Price ($)')
    
    plt.tight_layout()
    plt.savefig('actor_critic_agent_results.png')
    
    # Plot action distribution
    plt.figure(figsize=(12, 5))
    
    # Plot action types
    plt.subplot(1, 2, 1)
    action_labels = ['Buy', 'Sell', 'Hold']
    action_counts = [action_types.count(i) for i in range(3)]
    plt.bar(action_labels, action_counts)
    plt.title('Action Type Distribution')
    plt.ylabel('Count')
    
    # Plot action amounts
    plt.subplot(1, 2, 2)
    plt.hist(action_amounts, bins=10)
    plt.title('Action Amount Distribution')
    plt.xlabel('Amount')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('actor_critic_agent_actions.png')
    
    print(f"Final portfolio value: ${portfolio_values[-1]:.2f}")
    print(f"Initial portfolio value: ${portfolio_values[0]:.2f}")
    print(f"Return: {(portfolio_values[-1] / portfolio_values[0] - 1) * 100:.2f}%")
    
    return portfolio_values

if __name__ == "__main__":
    run_actor_critic_agent() 