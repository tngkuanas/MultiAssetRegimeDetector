from data_collection import *
from data_preprocessing import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class StrategyManager:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.models = {}
        self.strategies = {}

    def add_model(self, name, model_instance):
        """Register any ML/statistical model that supports fit() and predict()."""
        self.models[name] = model_instance

    def add_strategy(self, name, strategy_class):
        """Register any trading strategy class."""
        self.strategies[name] = strategy_class

    def run_model(self, model, data):
        """Train and predict using any model with .fit() and .predict()"""
        preprocessed_data = preprocess_data(data)
        X_train = preprocessed_data.iloc[:500][["Returns", "Range"]]
        X_test = preprocessed_data.iloc[500:][["Returns", "Range"]]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled)
        preds = model.predict(X_test_scaled)
        return preds, X_train, X_test

    def run_strategy(self, strategy_instance, data):
        """Run any strategy that has a .backtest() method."""
        # Expect generic method name for flexibility
        if hasattr(strategy_instance, "backtest"):
            strat_df, sharpe_b, sharpe_s = strategy_instance.backtest(data)
        elif hasattr(strategy_instance, "backtest_ma_crossover"):
            strat_df, sharpe_b, sharpe_s = strategy_instance.backtest_ma_crossover(12, 21, "long")
        else:
            raise AttributeError(f"{strategy_instance.__class__.__name__} has no backtest method.")
        return strat_df, sharpe_b, sharpe_s

    def run_all(self):
        """Run all registered models and strategies."""
        results = {}

        for model_name, model in self.models.items():
            print(f"\n🔹 Running model: {model_name}")

            from trading_strategy.moving_average_crossover import MovingAverageCrossover
            base_strategy = MovingAverageCrossover(self.symbol, self.start_date, self.end_date)
            strat_df, _, _ = base_strategy.backtest_ma_crossover(12, 21, "long")

            preds, X_train, X_test = self.run_model(model, strat_df)

            # Determine favourable regimes
            X_train["State"] = model.predict(StandardScaler().fit_transform(X_train))
            state_returns = X_train.groupby("State")["Returns"].mean()
            favourable_states = state_returns[state_returns > 0].index.tolist()

            strat_df["PSignal"] = 0
            strat_df.loc[X_test.index, "PSignal"] = [
                1 if s in favourable_states else 0 for s in preds
            ]

            for strat_name, strat_class in self.strategies.items():
                print(f"🔸 Strategy: {strat_name}")

                strategy = strat_class(self.symbol, self.start_date, self.end_date)

                # Allow strategy to take preprocessed data
                if hasattr(strategy, "change_df"):
                    strategy.change_df(strat_df)

                strat_df_2, sharpe_b, sharpe_s = self.run_strategy(strategy, strat_df)

                results[(model_name, strat_name)] = {
                    "Sharpe Benchmark": sharpe_b,
                    "Sharpe Strategy": sharpe_s,
                    "Returns Benchmark": strat_df_2["Bench_C_Rets"].iloc[-1],
                    "Returns Strategy": strat_df_2["Strat_C_Rets"].iloc[-1],
                }

                plt.figure(figsize=(12, 6))
                plt.plot(strat_df_2["Bench_C_Rets"], label="Benchmark")
                plt.plot(strat_df_2["Strat_C_Rets"], label=f"{model_name} + {strat_name}")
                plt.title(f"{model_name} + {strat_name} Performance")
                plt.legend()
                plt.show()

        return results
