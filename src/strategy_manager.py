import pandas as pd

class StrategyManager:
    def __init__(self):
        self.models = {}
        self.strategies = {}

    def add_model(self, name, model_instance):
        """Register any model instance."""
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
        Processes asset data to generate signals by coordinating models and strategies.
        """
        if not self.strategies:
            raise ValueError("No trading strategy registered. Please add a strategy using add_strategy().")

        asset_data_with_signals = asset_data.copy()
        
        # --- Model Signal Generation ---
        if self.models:
            # Assume one model for now
            model_name, model = next(iter(self.models.items()))
            
            # Check if the model has the standardized signal generation method
            if hasattr(model, 'generate_psignal'):
                print(f"Generating PSignal using model: {model_name}")
                psignal = model.generate_psignal(asset_data, macro_data)
                asset_data_with_signals['PSignal'] = psignal
            else:
                print(f"Warning: Model {model_name} does not have a 'generate_psignal' method. Defaulting PSignal to 1.")
                asset_data_with_signals["PSignal"] = 1
        else:
            # If no model is specified, default to a PSignal of 1 (always allow trading)
            asset_data_with_signals["PSignal"] = 1
        
        asset_data_with_signals['PSignal'] = asset_data_with_signals['PSignal'].fillna(0)
        
        # --- Strategy Signal Generation ---
        # Assume one strategy for now
        strategy_name, strategy_instance = next(iter(self.strategies.items()))
        
        print(f"Running strategy: {strategy_name}")
        asset_data_with_signals = self._run_strategy(strategy_instance, asset_data_with_signals, macro_data)
        
        asset_data_with_signals.dropna(inplace=True)
        return asset_data_with_signals

