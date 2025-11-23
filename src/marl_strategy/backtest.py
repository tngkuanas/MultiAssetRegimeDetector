import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.marl_strategy.environment import PortfolioEnv
from src.data_collection import get_data
from src.data_collection_fred import get_fred_data
from config import FRED_API_KEY

def backtest_marl_strategy():
    """
    Main function to load a trained PPO model and evaluate its performance.
    """
    # --- 1. Configuration ---
    # IMPORTANT: Ensure your FRED_API_KEY is set in config.py

    symbols = ["VOO", "NVO", "GOOG"]
    start_date = "2021-01-01"
    end_date = "2024-01-01"

    series_to_fetch = {
        'DGS10': '10y_treasury_yield',
        'T10Y2Y': '10y-2y_spread',
        'VIXCLS': 'vix'
    }

    window_size = 50
    model_load_path = "marl_ppo_portfolio_model.zip"

    if not os.path.exists(model_load_path):
        print(f"Error: Model file not found at '{model_load_path}'.")
        print("Please run the training script first (src/marl_strategy/train.py).")
        return

    # --- 2. Data Fetching for Backtest ---
    print("--- Fetching Data for Backtesting Environment ---")
    market_data = {}
    for symbol in symbols:
        market_data[symbol] = get_data(symbol, start_date, end_date)
    
    if FRED_API_KEY == 'YOUR_API_KEY_HERE' or not FRED_API_KEY:
        print("Warning: FRED API key not set in config.py. Skipping macroeconomic data.")
        macro_data = None
    else:
        macro_data = get_fred_data(series_ids=series_to_fetch, start_date=start_date, end_date=end_date)

    if not market_data or macro_data is None:
        print("Could not fetch all necessary data. Exiting backtest.")
        return

    # --- 3. Environment and Model Loading ---
    print("\n--- Initializing Backtesting Environment and Loading Model ---")
    
    # Create the environment instance first
    unwrapped_env = PortfolioEnv(market_data, macro_data, window_size=window_size)
    
    # Get necessary attributes BEFORE wrapping
    initial_portfolio_value = unwrapped_env.initial_portfolio_value
    # The number of steps will be max_steps + 1. The dates should match this length.
    num_backtest_steps = unwrapped_env.max_steps + 1
    dates = unwrapped_env.combined_data.index[window_size+1 : window_size+1+num_backtest_steps]

    # Now, wrap the environment for the model
    env = ss.pettingzoo_env_to_vec_env_v1(unwrapped_env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)
    
    model = PPO.load(model_load_path)

    # --- 4. Run Backtest Loop ---
    print("\n--- Running Backtest ---")
    obs = env.reset()
    done = False
    portfolio_values = [initial_portfolio_value]

    while len(portfolio_values) <= num_backtest_steps:
        actions, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)
        
        # 'infos' is a list of dicts in a vectorized env
        portfolio_values.append(infos[0]['portfolio_value'])

    env.close()

    # --- 5. Performance Analysis ---
    print("\n--- Backtest Complete. Analyzing Performance ---")
    # Ensure the length of portfolio_values matches the dates
    portfolio_df = pd.DataFrame({'Portfolio Value': portfolio_values[:len(dates)]}, index=dates)
    portfolio_df['Daily Returns'] = portfolio_df['Portfolio Value'].pct_change()
    portfolio_df['Cumulative Returns'] = (1 + portfolio_df['Daily Returns']).cumprod() - 1

    # Create a Buy and Hold benchmark
    raw_market_data = {}
    for symbol in symbols:
        raw_market_data[symbol] = get_data(symbol, start_date, end_date)
    
    close_prices = pd.DataFrame({s: df['Close'] for s, df in raw_market_data.items()})
    close_prices = close_prices.reindex(portfolio_df.index).ffill().bfill()
    benchmark_returns = close_prices.pct_change().mean(axis=1)
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1
    portfolio_df['Benchmark Cumulative Returns'] = benchmark_cumulative_returns

    final_strategy_return = portfolio_df['Cumulative Returns'].iloc[-1]
    final_benchmark_return = portfolio_df['Benchmark Cumulative Returns'].iloc[-1]
    
    def sharpe_ratio(return_series, annualization_factor=252):
        return np.sqrt(annualization_factor) * return_series.mean() / return_series.std()
        
    strategy_sharpe = sharpe_ratio(portfolio_df['Daily Returns'])
    benchmark_sharpe = sharpe_ratio(benchmark_returns)

    print("\n--- Performance Summary ---")
    print(f"MARL Strategy Final Cumulative Return: {final_strategy_return:.2%}")
    print(f"MARL Strategy Annualized Sharpe Ratio: {strategy_sharpe:.2f}")
    print("-" * 20)
    print(f"Benchmark Final Cumulative Return: {final_benchmark_return:.2%}")
    print(f"Benchmark Annualized Sharpe Ratio: {benchmark_sharpe:.2f}")

    # --- 6. Plotting ---
    plt.figure(figsize=(12, 6))
    portfolio_df['Cumulative Returns'].plot(label='MARL Strategy', legend=True)
    portfolio_df['Benchmark Cumulative Returns'].plot(label='Benchmark (Buy and Hold)', legend=True)
    plt.title("MARL Strategy Performance vs. Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    backtest_marl_strategy()
