import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from customenv import StockTradingEnv
from actor_critic_model import ActorCritic
import seaborn as sns

def random_policy(state, action_dim=2):
    """Simple random policy that outputs random actions"""
    action_type = np.random.randint(0, 3)  # 0: buy, 1: sell, 2: hold
    action_amount = np.random.random()  # Random amount between 0 and 1
    return np.array([action_type, action_amount])

def evaluate_policy(env, policy, num_episodes=5):
    """Evaluate a policy for multiple episodes"""
    episode_returns = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        episode_reward = 0
        portfolio_values = [info['portfolio_value']]
        
        while not done:
            if isinstance(policy, PPO):
                action, _ = policy.predict(state, deterministic=True)
            elif isinstance(policy, ActorCritic):
                action = policy.get_action(state, deterministic=True)
            else:  # Random policy
                action = policy(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            portfolio_values.append(info['portfolio_value'])
            state = next_state
        
        # Calculate returns as percentage change from initial value
        returns = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        episode_returns.append(returns)
    
    return np.mean(episode_returns), np.std(episode_returns)

def compare_policies():
    """Compare PPO, Actor-Critic and Random policies"""
    # Environment parameters
    env_params = {
        'initial_cash': 10000.0,
        'initial_stock_price': 100,
        'mu': 0.1,
        'sigma': 0.2,
        'risk_free_rate': 0.01,
        'max_steps': 252,  # One trading year
        'history_length': 30
    }
    
    # Create environment
    env = StockTradingEnv(**env_params)
    state_dim = env.observation_space.shape[0]
    action_dim = 2
    
    # Initialize policies
    ppo = PPO("MlpPolicy", env, verbose=1)
    actor_critic = ActorCritic(state_dim=state_dim, action_dim=action_dim)
    
    # Train policies
    print("Training PPO...")
    ppo.learn(total_timesteps=100000)
    
    print("Training Actor-Critic...")
    for episode in range(100):
        state, _ = env.reset()
        done = False
        while not done:
            action = actor_critic.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            actor_critic.store_transition(state, action, reward, next_state, done)
            state = next_state
        actor_critic.update_policy()
    
    # Evaluate policies
    policies = {
        'PPO': ppo,
        'Actor-Critic': actor_critic,
        'Random': random_policy
    }
    
    results = {}
    for name, policy in policies.items():
        mean_return, std_return = evaluate_policy(env, policy, num_episodes=10)
        results[name] = {'mean': mean_return, 'std': std_return}
    
    # Plot results
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    means = [results[name]['mean'] for name in names]
    stds = [results[name]['std'] for name in names]
    
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.title('Policy Performance Comparison')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig('policy_comparison.png')
    plt.close()

def analyze_ppo_sensitivity():
    """Analyze PPO's sensitivity to different parameters"""
    # Parameter ranges to test
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    mus = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3]
    sigmas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    
    # Results storage
    seed_results = []
    mu_results = []
    sigma_results = []
    
    # Base environment parameters
    base_params = {
        'initial_cash': 10000.0,
        'initial_stock_price': 100,
        'risk_free_rate': 0.01,
        'max_steps': 252,
        'history_length': 30
    }
    
    # Test different seeds
    print("Testing different seeds...")
    for seed in seeds:
        env = StockTradingEnv(**base_params, mu=0.1, sigma=0.2)
        ppo = PPO("MlpPolicy", env, verbose=1, seed=seed)
        ppo.learn(total_timesteps=50000)
        mean_return, _ = evaluate_policy(env, ppo)
        seed_results.append(mean_return)
    
    # Test different mus
    print("Testing different drift rates (mu)...")
    for mu in mus:
        env = StockTradingEnv(**base_params, mu=mu, sigma=0.2)
        ppo = PPO("MlpPolicy", env, verbose=1)
        ppo.learn(total_timesteps=50000)
        mean_return, _ = evaluate_policy(env, ppo)
        mu_results.append(mean_return)
    
    # Test different sigmas
    print("Testing different volatilities (sigma)...")
    for sigma in sigmas:
        env = StockTradingEnv(**base_params, mu=0.1, sigma=sigma)
        ppo = PPO("MlpPolicy", env, verbose=1)
        ppo.learn(total_timesteps=50000)
        mean_return, _ = evaluate_policy(env, ppo)
        sigma_results.append(mean_return)
    
    # Plot sensitivity analysis
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot seed sensitivity
    ax1.plot(range(len(seeds)), seed_results, 'o-')
    ax1.set_title('Seed Sensitivity')
    ax1.set_xlabel('Seed Index')
    ax1.set_ylabel('Return (%)')
    
    # Plot mu sensitivity
    ax2.plot(mus, mu_results, 'o-')
    ax2.set_title('Drift Rate (μ) Sensitivity')
    ax2.set_xlabel('μ')
    ax2.set_ylabel('Return (%)')
    
    # Plot sigma sensitivity
    ax3.plot(sigmas, sigma_results, 'o-')
    ax3.set_title('Volatility (σ) Sensitivity')
    ax3.set_xlabel('σ')
    ax3.set_ylabel('Return (%)')
    
    plt.tight_layout()
    plt.savefig('ppo_sensitivity.png')
    plt.close()

if __name__ == "__main__":
    print("Comparing policies...")
    compare_policies()
    
    print("Analyzing PPO sensitivity...")
    analyze_ppo_sensitivity() 