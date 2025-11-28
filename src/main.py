from portfolio_manager import PortfolioManager
from allocation_strategy import JumpModelRiskParityAllocationStrategy
from config_loader import load_config
from strategy_selector import get_portfolio_composition, select_strategy_version

if __name__ == "__main__":
    # --- Load Configuration ---
    config = load_config()
    symbols = config["symbols"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    fred_series = config["fred_series"]

    # --- Strategy Selection ---
    asset_definitions = config.get("asset_definitions", {})
    selection_rules = config.get("strategy_selection_rules", [])
    
    composition = get_portfolio_composition(symbols, asset_definitions)
    version_name = select_strategy_version(composition, selection_rules)
    
    print(f"\n--- Selected strategy version: '{version_name}' ---")
    sjm_params = config["strategy_versions"][version_name]
    # --- End Strategy Selection ---

    # --- Run Backtest ---
    print(f"\n--- RUNNING BACKTEST with version '{version_name}' ---")
    
    portfolio_manager = PortfolioManager(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        allocation_strategy=JumpModelRiskParityAllocationStrategy(
            hysteresis_period=sjm_params["hysteresis_period"],
            max_turnover=sjm_params["max_turnover"],
            vol_targets=sjm_params["vol_targets"],
            n_states=sjm_params["n_states"],
            jump_penalty=sjm_params["jump_penalty"],
        ),
        fred_series_to_fetch=fred_series,
    )
    portfolio_manager.run_portfolio_backtest(plot=True)