import pandas as pd
from data_preprocessing import preprocess_data
from sklearn.preprocessing import StandardScaler

class StrategyManager:
    def __init__(self):
        # Removed symbol, start_date, end_date as they will be handled by PortfolioManager
        self.models = {}
        self.strategies = {}

    def add_model(self, name, model_instance):
        """Register any ML/statistical model that supports fit() and predict()."""
        self.models[name] = model_instance

    def add_strategy(self, name, strategy_instance):
        """Register any trading strategy class instance with a generate_signals method."""
        self.strategies[name] = strategy_instance

    def _run_model(self, model, data):
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
    
    def _run_strategy(self, strategy_instance, data):
        """Run any strategy that has a .process() method."""
        if hasattr(strategy_instance, "process"):
            return strategy_instance.process(data)
        else:
            raise AttributeError(f"{strategy_instance.__class__.__name__} has no process method.")

    def process(self, asset_data):
        """
        Processes the asset data to generate signals.
        - Generates PSignal (regime filter) if a model is registered. Defaults to 1 if no model.
        - Generates Signal (trading action) using the registered strategy.
        Returns the asset_data DataFrame with 'PSignal' and 'Signal' columns added.
        """
        if not self.strategies:
            raise ValueError("No trading strategy registered. Please add a strategy using add_strategy().")

        asset_data_with_signals = asset_data.copy()
        strategy_name, strategy_instance = next(iter(self.strategies.items()))

        # Generate PSignal (Regime Filter) only if a model is registered
        if self.models:
            model_name, model = next(iter(self.models.items()))
            preds, X_train, X_test = self._run_model(model, asset_data_with_signals)

            X_train["State"] = model.predict(StandardScaler().fit_transform(X_train))
            state_returns = X_train.groupby("State")["Returns"].mean()
            favourable_states = state_returns[state_returns > 0].index.tolist()

            asset_data_with_signals["PSignal"] = 0
            psignal_dates = X_test.index
            psignal_values = [1 if s in favourable_states else 0 for s in preds]
            temp_psignal_series = pd.Series(psignal_values, index=psignal_dates)
            asset_data_with_signals.loc[psignal_dates, "PSignal"] = temp_psignal_series
        else:
            # If no model is registered, default PSignal to 1 (filter is always "on")
            asset_data_with_signals["PSignal"] = 1
        
        # Generate Signal (Trading Action)
        asset_data_with_signals = self._run_strategy(strategy_instance, asset_data_with_signals)

        asset_data_with_signals.dropna(inplace=True)
        return asset_data_with_signals
