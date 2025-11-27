from regime_detection.hidden_markov_model import HiddenMarkovModel
from trading_strategy.moving_average_crossover import MovingAverageCrossover
from strategy_manager import StrategyManager
from portfolio_manager import PortfolioManager
from allocation_strategy import (
    EqualWeightAllocationStrategy,
    ShuMulveyAllocationStrategy,
    JumpModelRiskParityAllocationStrategy,
)
from config_loader import load_config

if __name__ == "__main__":
    # --- Load Configuration ---
    config = load_config()
    symbols = config["symbols"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    fred_series = config["fred_series"]
    strategy_params = config["strategy_params"]

    # # --- STRATEGY 1: HMM + Moving Average Crossover with Equal Weighting ---
    # print("--- RUNNING STRATEGY 1: HMM + MA Crossover (Macro-Aware) ---")
    # signal_engine_1 = StrategyManager()
    # signal_engine_1.add_model("HMM", HiddenMarkovModel(n_components=4))
    # ma_params = strategy_params["moving_average_crossover"]
    # signal_engine_1.add_strategy(
    #     "MA_Crossover",
    #     MovingAverageCrossover(
    #         period_1=ma_params["period_1"],
    #         period_2=ma_params["period_2"],
    #         direction=ma_params["direction"],
    #     ),
    # )

    # portfolio_manager_1 = PortfolioManager(
    #     symbols=symbols,
    #     strategy_manager=signal_engine_1,
    #     start_date=start_date,
    #     end_date=end_date,
    #     allocation_strategy=EqualWeightAllocationStrategy(),
    #     fred_series_to_fetch=fred_series,
    # )
    # portfolio_manager_1.run_portfolio_backtest()

    # # --- STRATEGY 2: Shu & Mulvey GBDT Strategy ---
    # print("\n\n--- RUNNING STRATEGY 2: Shu & Mulvey GBDT Model ---")
    # signal_engine_2 = StrategyManager()

    # portfolio_manager_2 = PortfolioManager(
    #     symbols=symbols,
    #     strategy_manager=signal_engine_2,
    #     start_date=start_date,
    #     end_date=end_date,
    #     allocation_strategy=ShuMulveyAllocationStrategy(),
    #     fred_series_to_fetch=fred_series,
    # )
    # portfolio_manager_2.run_portfolio_backtest()

    # --- STRATEGY 3: Statistical Jump Model with Regime-Switched Risk Parity ---
    print("\n\n--- RUNNING STRATEGY 3: Statistical Jump Model (SJM) Risk Parity ---")
    signal_engine_3 = StrategyManager()
    sjm_params = strategy_params["jump_model_risk_parity"]

    portfolio_manager_3 = PortfolioManager(
        symbols=symbols,
        strategy_manager=signal_engine_3,
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