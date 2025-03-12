import gymnasium as gym
import numpy as np
import torch
import time
from customenv import StockTradingEnv
from actor_critic_model import ActorCritic

def train_actor_critic(env, model, num_episodes=1000, max_steps=1000):
    """
    Train the Actor-Critic model.
    
    Args:
        env: The environment to train in
        model: The Actor-Critic model
        num_episodes: Number of episodes to train for
        max_steps: Maximum steps per episode
        
    Returns:
        list: Episode rewards
    """
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Get action from model
            action = model.get_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            model.store_transition(state, action, reward, next_state, done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Train model after each step (online learning)
            model.update_policy()
            
            # Render environment every 10 steps
            if step % 10 == 0:
                env.render()
                time.sleep(0.1)  # Add a small delay to make rendering visible
                
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Portfolio Value: {info['portfolio_value']:.2f}")
    
    return episode_rewards

def evaluate_actor_critic(env, model, num_episodes=10, render=True):
    """
    Evaluate the trained Actor-Critic model.
    
    Args:
        env: The environment to evaluate in
        model: The trained Actor-Critic model
        num_episodes: Number of episodes to evaluate for
        render: Whether to render the environment
        
    Returns:
        float: Average episode reward
    """
    episode_rewards = []
    portfolio_returns = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nStarting evaluation episode {episode+1}/{num_episodes}")
        print(f"Initial portfolio value: {info['portfolio_value']:.2f}")
        
        while not done and step < 1000:
            # Get action from model (deterministic)
            with torch.no_grad():
                action = model.get_action(state)
            
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            step += 1
            
            # Render environment every 10 steps
            if render and step % 10 == 0:
                print(f"Step {step}, Action: {info['action_type']}, Amount: {info['action_amount']:.2f}, Reward: {reward:.4f}")
                env.render()
        
        episode_rewards.append(episode_reward)
        portfolio_returns.append(info["portfolio_return"])
        print(f"Evaluation Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Portfolio Return: {info['portfolio_return']:.2%}")
        print(f"Final portfolio value: {info['portfolio_value']:.2f}")
    
    avg_reward = np.mean(episode_rewards)
    avg_return = np.mean(portfolio_returns)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    print(f"Average Portfolio Return: {avg_return:.2%}")
    
    return avg_reward

if __name__ == "__main__":
    # Environment parameters
    stock_ticker = 'AAPL'
    starting_price = 100
    mu = 0.1
    sigma = 0.2
    risk_free_rate = 0.01
    horizon = 1000
    history_length = 30
    
    # Create environment
    env = StockTradingEnv(
        initial_cash=10000.0,
        initial_stock_price=starting_price,
        mu=mu,
        sigma=sigma,
        risk_free_rate=risk_free_rate,
        max_steps=horizon,
        history_length=history_length,
        render_mode='human'
    )
    
    # Calculate state dimension
    # Price history + portfolio info + holdings + returns
    state_dim = history_length + 4 + 1 + history_length
    
    # Action dimension: [action_type, amount]
    action_dim = 2
    
    # Create Actor-Critic model
    model = ActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99
    )
    
    # Train model
    print("Training Actor-Critic model...")
    train_actor_critic(env, model, num_episodes=25, max_steps=horizon)
    
    # Save model
    model.save("actor_critic_model.pt")
    
    # Evaluate model
    print("\nEvaluating Actor-Critic model...")
    evaluate_actor_critic(env, model, num_episodes=5, render=True)