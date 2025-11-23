from data_collection import *
from data_preprocessing import *
from regime_detection.hidden_markov_model import *
from trading_strategy.moving_average_crossover import *
from strategy_manager import *

if __name__ == "__main__":
    manager = StrategyManager(symbol="VOO", start_date="2004-01-01", end_date="2024-01-01")

    manager.add_model("HMM", HiddenMarkovModel())

    manager.add_strategy("MA_Crossover", MovingAverageCrossover)

    results = manager.run_all()

    print(results)

