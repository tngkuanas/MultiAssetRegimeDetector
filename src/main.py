from regime_detection.hidden_markov_model import HiddenMarkovModel
from trading_strategy.moving_average_crossover import MovingAverageCrossover
from trading_strategy.absolute_momentum import AbsoluteMomentum # New Import
from strategy_manager import StrategyManager
from portfolio_manager import PortfolioManager

if __name__ == "__main__":
    # --- Strategy Configuration ---
    # Define the portfolio assets and date range
    symbols = ["VOO", "NVO", "GOOG", "GLD", "VXUS"] # Expanded asset list for momentum strategy
    start_date = "2008-01-01"
    end_date = "2024-01-01"

    # --- STRATEGY 1: HMM + Moving Average Crossover with Equal Weighting ---
    print("--- RUNNING STRATEGY 1: HMM + MA Crossover ---")
    signal_engine_1 = StrategyManager()
    signal_engine_1.add_model("HMM", HiddenMarkovModel())
    signal_engine_1.add_strategy("MA_Crossover", MovingAverageCrossover())
    
    portfolio_manager_1 = PortfolioManager(
        symbols=symbols,
        strategy_manager=signal_engine_1,
        start_date=start_date,
        end_date=end_date,
        allocation_strategy="equal_weight"
    )
    portfolio_manager_1.run_portfolio_backtest()


    # # --- STRATEGY 2: Dual Momentum (Absolute + Relative Momentum) ---
    # print("\n\n--- RUNNING STRATEGY 2: Dual Momentum ---")
    # # 1. Configure the signal engine for Absolute Momentum
    # signal_engine_2 = StrategyManager()
    # signal_engine_2.add_strategy("Absolute_Momentum", AbsoluteMomentum(lookback_period=252))

    # # 2. Initialize the PortfolioManager with the 'relative_momentum' allocation strategy
    # portfolio_manager_2 = PortfolioManager(
    #     symbols=symbols,
    #     strategy_manager=signal_engine_2,
    #     start_date=start_date,
    #     end_date=end_date,
    #     allocation_strategy="relative_momentum"
    # )
    
    # # 3. Run the backtest
    # portfolio_manager_2.run_portfolio_backtest()

