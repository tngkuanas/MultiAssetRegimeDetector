from data_collection import *
from data_preprocessing import *
from regime_detection.hidden_markov_model import *
from trading_strategy.moving_average_crossover import *
from strategy_manager import *

if __name__ == "__main__":
    manager = StrategyManager(symbol="VOO", start_date="2004-01-01", end_date="2024-01-01")

    manager.add_model("HMM", HiddenMarkovModel())

    manager.add_strategy("MA_Crossover", MovingAverageCrossover)

    results = manager.run_all()

# ticker="VOO"
# start_date="2004-01-01"
# end_date="2024-01-01"
# strategy = MovingAverageCrossover(ticker, start_date, end_date)

# strat_df, sharpe_b, sharpe_s = strategy.backtest_ma_crossover(12, 21, "long", drop_cols=["High", "Low", "Volume"])
# strat_df


# # Structure Data
# X_train_2 = strat_df[["Returns", "Range"]].iloc[:500] # Train Test Split here
# X_test = strat_df[["Returns", "Range"]].iloc[500:]
# X_train_2.head()
# df_strat_mgr_test = strat_df.copy()
# len(X_train_2)

# from sklearn.preprocessing import StandardScaler
# X_train_2_scaled = StandardScaler().fit_transform(X_train_2)
# X_test_scaled = StandardScaler().fit_transform(X_test)

# # Fit Model
# hmm_model = GaussianHMM(n_components=4, covariance_type="full", n_iter=100).fit(X_train_2_scaled)
# print("Model Score:", hmm_model.score(X_train_2_scaled))

# # Predict Market Regimes
# hidden_states_preds = hmm_model.predict(X_test_scaled)
# hidden_states_preds[:10]
# len(hidden_states_preds)

# # Tag training data with hidden states
# train_states = hmm_model.predict(X_train_2_scaled)
# X_train_2["State"] = train_states

# # Compute mean returns per state
# state_returns = X_train_2.groupby("State")["Returns"].mean()
# print(state_returns)

# # Pick states with positive mean returns
# favourable_states = state_returns[state_returns > 0].index.tolist()
# print("Favourable states:", favourable_states)

# # Write Strategy
# state_signals = []
# for s in hidden_states_preds:
#     if s in favourable_states:
#         state_signals.append(1)
#     else:
#         state_signals.append(0)
# print("States: ", state_signals[:10])
# print("Lengh of States: ", len(state_signals))

# # Replace Strategy Dataframe
# df_strat_mgr_test = df_strat_mgr_test.tail(len(X_test_scaled))
# df_strat_mgr_test["PSignal"] = state_signals
# strategy.change_df(df_strat_mgr_test)
# strategy.df.head()

# strat_df_2, sharpe_b_2, sharpe_s_2 = strategy.backtest_ma_crossover(12, 21, "long")
# strat_df_2

# # Review equity curve
# print("Sharpe Ratio Benchmark: ", sharpe_b_2)
# print("Sharpe Ratio Regime Strategy with MA Cross: ", sharpe_s_2)
# print("--- ---")
# print(f"Returns Benchmark: {round(strat_df_2['Bench_C_Rets'].values[-1] * 100, 2)}%")
# print(f"Returns Regime Strategy with MA Cross: {round(strat_df_2['Strat_C_Rets'].values[-1] * 100, 2)}%")

# fig = plt.figure(figsize = (18, 10))
# plt.plot(strat_df_2["Bench_C_Rets"])
# plt.plot(strat_df_2["Strat_C_Rets"])
# plt.show()