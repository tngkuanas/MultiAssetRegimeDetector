import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Dict
from pettingzoo import ParallelEnv

class PortfolioEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "portfolio_v0"}

    def __init__(self, market_data, macro_data, window_size=50, initial_portfolio_value=100000, render_mode="human"):
        super().__init__()
        self.market_data = market_data
        self.macro_data = macro_data
        self.window_size = window_size
        self.symbols = list(market_data.keys())
        self.render_mode = render_mode
        self._prepare_data()

        self.initial_portfolio_value = initial_portfolio_value
        self.current_step = 0
        self.max_steps = len(self.combined_data) - self.window_size - 1

        self.possible_agents = self.symbols + ['CASH']
        self.agents = self.possible_agents[:]
        
        agent_observation_space = Dict({
            "market": Box(low=-np.inf, high=np.inf, shape=(self.window_size, 5), dtype=np.float32),
            "macro": Box(low=-np.inf, high=np.inf, shape=(len(self.macro_data.columns),), dtype=np.float32)
        })
        
        agent_action_space = Box(low=-5, high=5, shape=(1,), dtype=np.float32)

        # Make observation and action spaces identical for all agents as required by SB3 + supersuit
        self.observation_spaces = {agent: agent_observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: agent_action_space for agent in self.possible_agents}

    def _prepare_data(self):
        # Keep raw prices for calculating returns later
        self.raw_close_prices = pd.DataFrame({s: df['Close'] for s, df in self.market_data.items()})

        # --- Refactored Data Flattening ---
        # Rename columns before concatenating to avoid MultiIndex
        dfs_to_concat = []
        for symbol, df in self.market_data.items():
            renamed_df = df.rename(columns={c: f"{symbol}_{c}" for c in df.columns})
            dfs_to_concat.append(renamed_df)

        # Create a single large DataFrame with unique column names
        panel = pd.concat(dfs_to_concat, axis=1)

        # Now, joining with macro_data (which has a single-level index) is straightforward
        self.combined_data = panel.join(self.macro_data)
        # --- End Refactoring ---

        self.combined_data.ffill(inplace=True)
        self.combined_data.bfill(inplace=True)

        # Normalize all columns
        for col in self.combined_data.columns:
            # Use pct_change for price/volume data, z-score for others
            if any(price_col in col for price_col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                self.combined_data[col] = self.combined_data[col].pct_change().fillna(0)
            else: # Normalize macro data and any other indicators
                self.combined_data[col] = (self.combined_data[col] - self.combined_data[col].mean()) / self.combined_data[col].std()
        
        self.combined_data.replace([np.inf, -np.inf], 0, inplace=True)
        self.raw_close_prices = self.raw_close_prices.reindex(self.combined_data.index).ffill().bfill()


    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.portfolio_value = self.initial_portfolio_value
        self.portfolio_weights = {agent: 1.0 / len(self.agents) for agent in self.agents}

        observations = self._get_observations()
        infos = self._get_infos()
        
        return observations, infos

    def step(self, actions):
        if self.current_step >= self.max_steps:
            return self._get_observations(), {agent: 0 for agent in self.agents}, \
                   {agent: True for agent in self.agents}, {agent: False for agent in self.agents}, \
                   self._get_infos()

        raw_logits = np.array([actions[agent][0] for agent in self.agents])
        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        normalized_weights = exp_logits / np.sum(exp_logits)
        
        current_weights = {agent: normalized_weights[i] for i, agent in enumerate(self.agents)}

        current_prices = self.raw_close_prices.iloc[self.current_step + self.window_size]
        next_prices = self.raw_close_prices.iloc[self.current_step + self.window_size + 1]
        
        step_returns = (next_prices / current_prices) - 1
        
        portfolio_return = 0
        for agent in self.symbols:
            portfolio_return += current_weights[agent] * step_returns[agent]

        previous_portfolio_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)

        reward = np.log(self.portfolio_value) - np.log(previous_portfolio_value)
        rewards = {agent: reward for agent in self.agents}

        self.current_step += 1
        self.portfolio_weights = current_weights
        
        observations = self._get_observations()
        terminations = {agent: self.current_step >= self.max_steps for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = self._get_infos()

        return observations, rewards, terminations, truncations, infos

    def _get_observations(self):
        observations = {}
        current_data_slice = self.combined_data.iloc[self.current_step : self.current_step + self.window_size]
        current_macro_data = current_data_slice[self.macro_data.columns].values[-1,:].astype(np.float32)

        for agent in self.symbols:
            # Select columns using the new flattened names
            market_cols = [f"{agent}_Open", f"{agent}_High", f"{agent}_Low", f"{agent}_Close", f"{agent}_Volume"]
            market_obs = current_data_slice[market_cols].values.astype(np.float32)
            observations[agent] = {"market": market_obs, "macro": current_macro_data}

        observations['CASH'] = {
            "market": np.zeros((self.window_size, 5), dtype=np.float32),
            "macro": current_macro_data
        }
        return observations

    def _get_infos(self):
        return {agent: {"portfolio_value": self.portfolio_value, "weights": self.portfolio_weights} for agent in self.agents}

    def render(self, mode="human"):
        if mode == "human":
            print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.2f}")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
