# Quantitative Backtesting Engine

This project provides a flexible framework for backtesting various quantitative trading strategies on financial assets. It supports data collection, preprocessing, modular strategy implementation, and advanced performance evaluation against multiple benchmarks.

## Features

*   **Modular Strategy Design:** Easily integrate new trading strategies and regime detection models.
*   **Dynamic Backtesting:** Supports walk-forward backtesting with configurable rebalancing periods.
*   **Comprehensive Performance Metrics:** Evaluates strategies using Cumulative Returns, Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.
*   **Multiple Benchmarks:** Compares strategy performance against:
    *   Static Risk Parity (Monthly Rebalanced)
    *   Monthly Rebalanced Equal Weight
*   **Data Integration:** Utilizes `yfinance` for market data and `fredapi` for macroeconomic indicators.

## Setup and Installation

To get the project up and running, follow these steps:

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <your-repository-url>
    # cd DSP-Anas
    ```

2.  **Create a Python Virtual Environment (Recommended):**
    A virtual environment isolates project dependencies, preventing conflicts with other Python projects.

    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    Install all required Python libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration and Usage

The main configuration for running backtests is done directly within the `src/main.py` file.

1.  **Open `src/main.py`:** Use your preferred text editor or IDE to open this file.

2.  **Configure Assets (`symbols` list):**
    Locate the `symbols` variable near the top of the `src/main.py` file. You can modify this list to include the stock tickers you wish to backtest. For example:

    ```python
    # --- General Configuration ---
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN"] # Your custom list of tickers
    ```
    *Make sure the tickers are valid symbols recognized by Yahoo Finance.*

3.  **Configure Date Range:**
    Adjust the `start_date` and `end_date` variables to define the period for your backtest.

    ```python
    start_date = "2008-01-01"
    end_date = "2024-01-01"
    ```

4.  **Configure FRED API Key (if using macroeconomic data):**
    Some strategies and data collection modules (e.g., `data_collection_fred.py`) require a FRED API key.
    1.  Obtain your free API key from the FRED website: [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
    2.  Open the `config.py` file in the project's root directory and replace `'YOUR_FRED_API_KEY_HERE'` with your actual key:
        ```python
        FRED_API_KEY = 'YOUR_API_KEY_HERE' # Replace with your key
        ```
        *Note: `config.py` is typically in `.gitignore` to prevent committing your personal API key.*

5.  **Select Strategy 3 (Statistical Jump Model with Regime-Switched Risk Parity):**
    The `src/main.py` file contains commented-out blocks for different strategies. To run only Strategy 3:
    *   Ensure the block labeled `--- STRATEGY 3: Statistical Jump Model with Regime-Switched Risk Parity ---` is **uncommented**.
    *   Ensure all other strategy blocks (e.g., "STRATEGY 1: HMM + Moving Average Crossover", "STRATEGY 2: Shu & Mulvey GBDT Strategy") are **commented out**.

    A correctly configured section for Strategy 3 should look like this (other sections should have `#` at the beginning of each line):

    ```python
    # ... (other code) ...

    # --- STRATEGY 3: Statistical Jump Model with Regime-Switched Risk Parity ---
    print("\n\n--- RUNNING STRATEGY 3: Statistical Jump Model (SJM) Risk Parity ---")
    signal_engine_3 = StrategyManager()

    portfolio_manager_3 = PortfolioManager(
        symbols=symbols,
        strategy_manager=signal_engine_3,
        start_date=start_date,
        end_date=end_date,
        allocation_strategy="jump_model_risk_parity",
        fred_series_to_fetch=fred_series
    )
    portfolio_manager_3.run_portfolio_backtest()

    # ... (rest of the file, other strategy blocks commented out) ...
    ```

## Running a Backtest

Once configured, execute the `main.py` script from your project's root directory:

```bash
python src/main.py
```

## Interpreting the Results

Upon successful execution, the script will:
*   Print a detailed performance summary to the console, including Cumulative Returns, Sharpe Ratio, Sortino Ratio, and Maximum Drawdown for your selected strategy and the two benchmarks.
*   Display a `matplotlib` plot showing the cumulative returns of your strategy against the benchmarks. Close the plot window to terminate the script.

## Customizing Strategy Parameters

Advanced users can modify the parameters of `StatisticalJumpModel` (e.g., `jump_penalty`, `n_states`) or the allocation strategy (e.g., `hysteresis_period`, `max_turnover`) by adjusting the instantiation calls within `src/portfolio_manager.py` or `src/main.py` as appropriate.
