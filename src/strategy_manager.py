import pandas as pd
from data_preprocessing import preprocess_data
from sklearn.preprocessing import StandardScaler

class StrategyManager:
    def __init__(self):
        self.models = {}
        self.strategies = {}

    def add_model(self, name, model_instance):
        """Register any ML/statistical model that supports fit() and predict()."""
        self.models[name] = model_instance

    def add_strategy(self, name, strategy_instance):
        """Register any trading strategy class instance with a process method."""
        self.strategies[name] = strategy_instance

    def _run_strategy(self, strategy_instance, data, macro_data):
        """Run any strategy that has a .process() method."""
        if hasattr(strategy_instance, "process"):
            return strategy_instance.process(data, macro_data)
        else:
            raise AttributeError(f"{strategy_instance.__class__.__name__} has no process method.")

    def process(self, asset_data, macro_data=None):
        """
        Processes asset data to generate signals, incorporating macro data if available.
        """
        if not self.strategies:
            raise ValueError("No trading strategy registered. Please add a strategy using add_strategy().")

        asset_data_with_signals = asset_data.copy()
        strategy_name, strategy_instance = next(iter(self.strategies.items()))

        if self.models:
            model_name, model = next(iter(self.models.items()))
            
            # 1. Preprocess data by merging market and macro data
            preprocessed_data = preprocess_data(asset_data_with_signals)
            if macro_data is not None:
                preprocessed_data = preprocessed_data.join(macro_data)
                preprocessed_data.ffill(inplace=True)
                preprocessed_data.dropna(inplace=True)
            
            X_train = preprocessed_data.iloc[:500]
            X_test = preprocessed_data.iloc[500:]

            # 2. Fit and Predict based on model type
            if hasattr(model, 'regime_labels'): # Heuristic for JumpAwareModel
                model.fit(X_train)
                preds = model.predict(X_test)
                asset_data_with_signals.loc[preds.index, "PSignal"] = preds
            else: # Assumed to be HMM or similar numerical model
                hmm_features = ['Returns', 'Range']
                if macro_data is not None:
                    hmm_features.extend(macro_data.columns)
                
                # Ensure all features are present in X_train/X_test before using them
                hmm_features = [f for f in hmm_features if f in X_train.columns]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train[hmm_features])
                model.fit(X_train_scaled) # The one and only fit call for HMM

                # Predict on the test set for PSignal
                X_test_scaled = scaler.transform(X_test[hmm_features])
                test_preds = model.predict(X_test_scaled)
                
                # Determine favourable states by predicting on the training set
                train_preds = model.predict(X_train_scaled)
                X_train_with_states = X_train.copy()
                X_train_with_states['State'] = train_preds
                state_returns = X_train_with_states.groupby('State')["Returns"].mean()
                favourable_states = state_returns[state_returns > 0].index.tolist()
                
                psignal_series = pd.Series(test_preds, index=X_test.index).apply(lambda x: 1 if x in favourable_states else 0)
                asset_data_with_signals.loc[psignal_series.index, "PSignal"] = psignal_series
        else:
            asset_data_with_signals["PSignal"] = 1
        
        asset_data_with_signals['PSignal'].fillna(0, inplace=True)
        
        asset_data_with_signals = self._run_strategy(strategy_instance, asset_data_with_signals, macro_data)
        asset_data_with_signals.dropna(inplace=True)
        return asset_data_with_signals
