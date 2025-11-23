# User Guideline for the DSP-Anas Backtesting Project

This document provides a guide for using and extending the quantitative trading strategy backtesting framework.

## Project Overview

This project is a flexible framework for backtesting various trading strategies on financial assets. It supports:
*   Data collection from `yfinance` for market data and `fredapi` for macroeconomic indicators.
*   Advanced data preprocessing.
*   Modular strategy and model implementation (e.g., Hidden Markov Models, Moving Average Crossover, Dual Momentum).
*   Multiple portfolio allocation strategies.
*   A sophisticated Multi-Agent Reinforcement Learning (MARL) backtesting pipeline.

---

## 1. Installation

To ensure all necessary libraries are installed, please run the following command in your project's root directory:

```bash
pip install -r requirements.txt
```

---

## 2. Using the Rules-Based Strategies

The `src/main.py` file is configured to run rules-based strategies. You can easily switch between different strategies by uncommenting the desired strategy block.

### 2.1 Dual Momentum Strategy (Default)

This strategy combines Absolute Momentum (trend following to avoid downturns) with Relative Momentum (picking the best performer among trending assets).

*   **Configuration:**
    *   Strategy: `AbsoluteMomentum` (defined in `src/trading_strategy/absolute_momentum.py`)
    *   Allocation: `"relative_momentum"` (defined in `src/portfolio_manager.py`)
*   **To Run:**
    ```bash
    python src/main.py
    ```
    This is the default uncommented strategy in `main.py`.

### 2.2 HMM + Moving Average Crossover Strategy

This strategy uses a Hidden Markov Model to identify market regimes and filters a Moving Average Crossover strategy based on "favorable" regimes.

*   **Configuration:**
    *   Model: `HiddenMarkovModel` (defined in `src/regime_detection/hidden_markov_model.py`)
    *   Strategy: `MovingAverageCrossover` (defined in `src/trading_strategy/moving_average_crossover.py`)
    *   Allocation: `"equal_weight"` (defined in `src/portfolio_manager.py`)
*   **To Run:**
    1.  Open `src/main.py`.
    2.  **Comment out** the `--- STRATEGY 2: Dual Momentum ---` block.
    3.  **Uncomment** the `--- STRATEGY 1: HMM + Moving Average Crossover with Equal Weighting ---` block.
    4.  Save `main.py` and then run:
        ```bash
        python src/main.py
        ```

---

## 3. Multi-Agent Reinforcement Learning (MARL) Strategy

This is an advanced, AI-driven approach where multiple agents learn to manage a portfolio.

### 3.1 FRED API Key Requirement

The MARL environment uses macroeconomic data from the Federal Reserve (FRED). You need a free API key:

1.  Obtain your API key from the FRED website: [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2.  **Paste your API key** into the `FRED_API_KEY` variable in these files:
    *   `src/data_collection_fred.py`
    *   `src/marl_strategy/train.py`
    *   `src/marl_strategy/backtest.py`

### 3.2 Training the MARL Model

Training is computationally intensive and can take a long time, especially without a GPU.

*   **To Run Training:**
    ```bash
    python src/marl_strategy/train.py
    ```
*   **Output:** Upon successful completion, a file named `marl_portfolio_model.zip` will be created in your project's root directory. This file contains the trained agents' policies.

### 3.3 Backtesting the MARL Strategy

Once you have a trained model, you can evaluate its performance on unseen data.

*   **To Run Backtest:**
    ```bash
    python src/marl_strategy/backtest.py
    ```
*   **Output:** This will provide a performance summary and a plot comparing the MARL strategy's cumulative returns against a Buy and Hold benchmark.

---

## 4. Extensibility

The framework is designed to be highly modular.

*   **Adding New Regime Models:** Create a new class (e.g., in `src/regime_detection/`) with `.fit()` and `.predict()` methods, then register it with `StrategyManager.add_model()`.
*   **Adding New Trading Strategies:** Create a new class (e.g., in `src/trading_strategy/`) with a `process(data)` method that returns the DataFrame with a `Signal` column, then register it with `StrategyManager.add_strategy()`.
*   **Adding New Allocation Strategies:** Implement a new `_determine_[your_strategy_name]_weights()` method in `src/portfolio_manager.py` and update the `allocation_strategy` parameter to include its name.
*   **Adding New MARL Environments/Algorithms:** Extend the `src/marl_strategy/` directory with new environments or integrate different MARL algorithms.

---

## 5. Important Notes

*   **Corrected Returns Calculation:** The `data_preprocessing.py` file has been updated to use the standard log returns calculation (`data["Log"].diff()`). This was a critical fix.
*   **Interpreting Backtest Results:** Strategies that aim to reduce risk by going to cash (like Dual Momentum or HMM-filtered strategies) may show lower returns and Sharpe Ratios compared to an "always-on" Buy and Hold benchmark, especially during sustained bull markets. Their true value often lies in mitigating losses during downturns.
*   **`__pycache__` Directories:** These directories contain compiled Python files and are automatically generated. You can safely ignore or delete them.

---

We hope this guideline helps you effectively use and expand your backtesting framework!
