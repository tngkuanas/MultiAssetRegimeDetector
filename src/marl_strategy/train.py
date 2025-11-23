import os
import sys
import torch as th
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

def train_marl_strategy():
    """
    Main function to set up and run the MARL training process using PPO.
    """
    # --- 1. Configuration ---
    # IMPORTANT: Ensure your FRED_API_KEY is set in config.py
    
    symbols = ["VOO", "NVO", "GOOG"]
    start_date = "2000-01-01"
    end_date = "2020-12-31"

    series_to_fetch = {
        'DGS10': '10y_treasury_yield',
        'T10Y2Y': '10y-2y_spread',
        'VIXCLS': 'vix'
    }

    total_timesteps = 250000 # Increased timesteps for better learning
    window_size = 50
    model_save_path = "marl_ppo_portfolio_model.zip"

    # --- 2. Data Fetching ---
    print("--- Fetching Data for Training Environment ---")
    market_data = {}
    for symbol in symbols:
        market_data[symbol] = get_data(symbol, start_date, end_date)
    
    if FRED_API_KEY == 'YOUR_API_KEY_HERE' or not FRED_API_KEY:
        print("Warning: FRED API key not set in config.py. Skipping macroeconomic data.")
        macro_data = None
    else:
        macro_data = get_fred_data(series_ids=series_to_fetch, start_date=start_date, end_date=end_date)

    if not market_data or macro_data is None:
        print("Could not fetch all necessary data. Exiting training.")
        return

    # --- 3. Environment Setup ---
    print("\n--- Initializing and Wrapping MARL Environment for SB3 ---")
    env = PortfolioEnv(market_data, macro_data, window_size=window_size)
    # Wrap the PettingZoo environment with supersuit to make it compatible with SB3
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)

    # --- 4. Model Training ---
    print(f"\n--- Starting PPO Training for {total_timesteps} Timesteps ---")
    
    # Check for CUDA availability
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Instantiate the PPO model.
    # "MultiInputPolicy" is used for environments with Dict observation spaces.
    model = PPO(
        "MultiInputPolicy",
        env,
        tensorboard_log="./marl_tensorboard_logs/",
        verbose=1,
        device=device
    )

    # Run the training loop
    model.learn(total_timesteps=total_timesteps)

    # --- 5. Save the Trained Model ---
    print(f"\n--- Training Complete. Saving model to {model_save_path} ---")
    model.save(model_save_path)
    env.close()
    print("Model saved successfully.")


if __name__ == '__main__':
    if not os.path.exists("marl_tensorboard_logs"):
        os.makedirs("marl_tensorboard_logs")
        
    train_marl_strategy()
