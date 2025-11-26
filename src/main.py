from regime_detection.hidden_markov_model import HiddenMarkovModel
from trading_strategy.moving_average_crossover import MovingAverageCrossover
from strategy_manager import StrategyManager
from portfolio_manager import PortfolioManager

if __name__ == "__main__":
    # --- General Configuration ---
    symbols = ["VOO","VXUS","GLD","BND","META"] 
    start_date = "2000-01-01" # Start later to ensure enough data for features
    end_date = "2024-01-01"

    # Define macroeconomic indicators to use across all strategies
    # Expanded for the Shu-Mulvey model
    fred_series = {
        'DGS10': '10y_treasury_yield',
        'T10Y2Y': '10y-2y_spread',
        'VIXCLS': 'vix',
        'CPIAUCSL': 'cpi',
        'UNRATE': 'unemployment_rate',
        'INDPRO': 'industrial_production', # Growth
        'TEDRATE': 'ted_spread' # Liquidity
    }

    # # --- STRATEGY 1: HMM + Moving Average Crossover with Equal Weighting ---
    # print("--- RUNNING STRATEGY 1: HMM + MA Crossover (Macro-Aware) ---")
    # signal_engine_1 = StrategyManager()
    # signal_engine_1.add_model("HMM", HiddenMarkovModel(n_components=4)) # Using 4 states for more nuance
    # signal_engine_1.add_strategy("MA_Crossover", MovingAverageCrossover())
    
    # portfolio_manager_1 = PortfolioManager(
    #     symbols=symbols,
    #     strategy_manager=signal_engine_1,
    #     start_date=start_date,
    #     end_date=end_date,
    #     allocation_strategy="equal_weight",
    #     fred_series_to_fetch=fred_series
    # )
    # portfolio_manager_1.run_portfolio_backtest()


    # # --- STRATEGY 2: Shu & Mulvey GBDT Strategy ---
    # print("\n\n--- RUNNING STRATEGY 2: Shu & Mulvey GBDT Model ---")
    # # This strategy's logic is self-contained within the PortfolioManager,
    # # so we can use a blank StrategyManager.
    # signal_engine_2 = StrategyManager()

    # portfolio_manager_2 = PortfolioManager(
    #     symbols=symbols,
    #     strategy_manager=signal_engine_2,
    #     start_date=start_date,
    #     end_date=end_date,
    #     allocation_strategy="shu_mulvey", # Use the new allocation strategy
    #     fred_series_to_fetch=fred_series
    # )
    # portfolio_manager_2.run_portfolio_backtest()

    # --- STRATEGY 3: Statistical Jump Model with Regime-Switched Risk Parity ---
    print("\n\n--- RUNNING STRATEGY 3: Statistical Jump Model (SJM) Risk Parity ---")
    # This strategy's logic is self-contained within the PortfolioManager,
    # so we can use a blank StrategyManager.
    signal_engine_3 = StrategyManager()

    portfolio_manager_3 = PortfolioManager(
        symbols=symbols,
        strategy_manager=signal_engine_3,
        start_date=start_date,
        end_date=end_date,
        allocation_strategy="jump_model_risk_parity", # Use the new allocation strategy
        fred_series_to_fetch=fred_series
    )
    portfolio_manager_3.run_portfolio_backtest()