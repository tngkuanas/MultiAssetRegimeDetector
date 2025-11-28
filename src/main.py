from portfolio_manager import PortfolioManager
from allocation_strategy import JumpModelRiskParityAllocationStrategy
from config_loader import load_config

if __name__ == "__main__":
    # --- Load Configuration ---
    config = load_config()
    symbols = config["symbols"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    fred_series = config["fred_series"]
    strategy_params = config["strategy_params"]

    # --- STRATEGY 3: Statistical Jump Model with Regime-Switched Risk Parity ---
    print("\n\n--- RUNNING STRATEGY 3: Statistical Jump Model (SJM) Risk Parity ---")
    sjm_params = strategy_params["jump_model_risk_parity"]

    portfolio_manager_3 = PortfolioManager(
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
    portfolio_manager_3.run_portfolio_backtest()