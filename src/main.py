from regime_detection.hidden_markov_model import HiddenMarkovModel
from regime_detection.jump_aware_model import JumpAwareModel
from trading_strategy.moving_average_crossover import MovingAverageCrossover
from trading_strategy.absolute_momentum import AbsoluteMomentum
from strategy_manager import StrategyManager
from portfolio_manager import PortfolioManager

if __name__ == "__main__":
    # --- General Configuration ---
    symbols = ["SPY", "QQQ", "BND", "TLT", "GLD"] 
    start_date = "2007-01-01"
    end_date = "2024-01-01"

    # Define macroeconomic indicators to use across all strategies
    fred_series = {
        'DGS10': '10y_treasury_yield',
        'T10Y2Y': '10y-2y_spread',
        'VIXCLS': 'vix'
    }

    # --- STRATEGY 1: HMM + Moving Average Crossover with Equal Weighting (Now with Macro Data) ---
    print("--- RUNNING STRATEGY 1: HMM + MA Crossover (Macro-Aware) ---")
    signal_engine_1 = StrategyManager()
    signal_engine_1.add_model("HMM", HiddenMarkovModel(n_components=4)) # Using 4 states for more nuance
    signal_engine_1.add_strategy("MA_Crossover", MovingAverageCrossover())
    
    portfolio_manager_1 = PortfolioManager(
        symbols=symbols,
        strategy_manager=signal_engine_1,
        start_date=start_date,
        end_date=end_date,
        allocation_strategy="equal_weight",
        fred_series_to_fetch=fred_series
    )
    portfolio_manager_1.run_portfolio_backtest()


    # --- STRATEGY 2: Dual Momentum (Absolute + Relative Momentum) (Now with Macro Data) ---
    print("\n\n--- RUNNING STRATEGY 2: Dual Momentum (Macro-Aware) ---")
    signal_engine_2 = StrategyManager()
    signal_engine_2.add_strategy("Absolute_Momentum", AbsoluteMomentum(lookback_period=252))
    
    portfolio_manager_2 = PortfolioManager(
        symbols=symbols,
        strategy_manager=signal_engine_2,
        start_date=start_date,
        end_date=end_date,
        allocation_strategy="relative_momentum",
        fred_series_to_fetch=fred_series
    )
    portfolio_manager_2.run_portfolio_backtest()


    # --- STRATEGY 3: MSJD (Approximate) + Crash-Aware Allocation (Now with Macro Data) ---
    print("\n\n--- RUNNING STRATEGY 3: Jump-Aware Model + Crash-Aware Allocation ---")
    signal_engine_3 = StrategyManager()
    signal_engine_3.add_model("JumpAware", JumpAwareModel())
    signal_engine_3.add_strategy("MA_Crossover", MovingAverageCrossover()) 
    
    portfolio_manager_3 = PortfolioManager(
        symbols=symbols,
        strategy_manager=signal_engine_3,
        start_date=start_date,
        end_date=end_date,
        allocation_strategy="crash_aware",
        fred_series_to_fetch=fred_series
    )
    portfolio_manager_3.run_portfolio_backtest()


