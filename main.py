import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from customenv import StockTradingEnv
from sb3_contrib import RecurrentPPO




if __name__=="__main__":
    print("Hello, World!")
    stock_ticker = 'AAPL'
    starting_price = 100
    mu = 0.1
    sigma = 0.2
    risk_free_rate = 0.01
    horizon = 1000
    state_dim = 10
    action_dim = 2
    
    cash = 1000
    env = StockTradingEnv(cash,starting_price,mu, sigma, risk_free_rate, horizon, history_length=30, render_mode='human')

    # model = PPO("MlpPolicy", env, verbose=1)
    # model.learn(total_timesteps=100000, log_interval=10)

    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
    model.learn(100000)

    model_env = model.get_env()
    obs = model_env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = model_env.step(action)
        model_env.render()

    
