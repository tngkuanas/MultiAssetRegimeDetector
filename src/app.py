from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict, Any, Optional
import uuid
import time

# Import your existing backtesting logic
from config_loader import load_config
from portfolio_manager import PortfolioManager
from allocation_strategy import JumpModelRiskParityAllocationStrategy
from strategy_selector import get_portfolio_composition, select_strategy_version

app = FastAPI()

# --- In-Memory Storage for Job Results ---
results_store: Dict[str, Dict[str, Any]] = {}


# --- Pydantic Models for API ---

class BacktestSummary(BaseModel):
    strategy_final_return: float
    strategy_sharpe_ratio: float
    strategy_sortino_ratio: float
    strategy_max_drawdown: float
    risk_parity_final_return: float
    risk_parity_sharpe_ratio: float
    risk_parity_sortino_ratio: float
    risk_parity_max_drawdown: float
    equal_weight_final_return: float
    equal_weight_sharpe_ratio: float
    equal_weight_sortino_ratio: float
    equal_weight_max_drawdown: float

class BacktestResult(BaseModel):
    summary: BacktestSummary
    cumulative_returns: list[Dict[str, Any]]

class Job(BaseModel):
    job_id: str
    status: str
    result: Optional[BacktestResult] = None
    error: Optional[str] = None


# --- Helper Function to run the backtest ---

def run_backtest_and_store_results(job_id: str):
    """
    A wrapper function that runs the backtest and stores the result.
    This function is designed to be run in the background.
    """
    try:
        print(f"Starting backtest for job_id: {job_id}")
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
        
        print(f"Selected strategy version: '{version_name}'")
        sjm_params = config["strategy_versions"][version_name]
        # --- End Strategy Selection ---

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
        
        portfolio_returns, _ = portfolio_manager.run_portfolio_backtest()

        # --- Format the results ---
        summary = BacktestSummary(
            strategy_final_return=portfolio_returns['Strategy Cumulative'].iloc[-1],
            strategy_sharpe_ratio=portfolio_manager.sharpe_ratio(portfolio_returns['Strategy Daily']),
            strategy_sortino_ratio=portfolio_manager.sortino_ratio(portfolio_returns['Strategy Daily']),
            strategy_max_drawdown=portfolio_manager.max_drawdown(portfolio_returns['Strategy Cumulative']),
            risk_parity_final_return=portfolio_returns['Risk Parity Cumulative'].iloc[-1],
            risk_parity_sharpe_ratio=portfolio_manager.sharpe_ratio(portfolio_returns['Risk Parity Daily']),
            risk_parity_sortino_ratio=portfolio_manager.sortino_ratio(portfolio_returns['Risk Parity Daily']),
            risk_parity_max_drawdown=portfolio_manager.max_drawdown(portfolio_returns['Risk Parity Cumulative']),
            equal_weight_final_return=portfolio_returns['Equal Weight Rebalanced Cumulative'].iloc[-1],
            equal_weight_sharpe_ratio=portfolio_manager.sharpe_ratio(portfolio_returns['Equal Weight Rebalanced Daily']),
            equal_weight_sortino_ratio=portfolio_manager.sortino_ratio(portfolio_returns['Equal Weight Rebalanced Daily']),
            equal_weight_max_drawdown=portfolio_manager.max_drawdown(portfolio_returns['Equal Weight Rebalanced Cumulative']),
        )

        chart_data = portfolio_returns[['Strategy Cumulative', 'Risk Parity Cumulative', 'Equal Weight Rebalanced Cumulative']].reset_index()
        chart_data['Date'] = chart_data['Date'].dt.strftime('%Y-%m-%d')
        
        result_data = BacktestResult(
            summary=summary,
            cumulative_returns=chart_data.to_dict(orient='records')
        )
        
        results_store[job_id].update({"status": "completed", "result": result_data})
        print(f"Completed backtest for job_id: {job_id}")

    except Exception as e:
        print(f"Error running backtest for job_id: {job_id}. Error: {e}")
        results_store[job_id].update({"status": "error", "error": str(e)})


# --- API Endpoints ---

@app.post("/run_strategy_3", response_model=Job)
def run_backtest_endpoint(background_tasks: BackgroundTasks):
    """
    Kicks off the backtest for Strategy 3 in the background.
    """
    job_id = str(uuid.uuid4())
    results_store[job_id] = {
        "job_id": job_id, 
        "status": "in_progress", 
        "result": None, 
        "error": None
    }
    background_tasks.add_task(run_backtest_and_store_results, job_id)
    return results_store[job_id]

@app.get("/results/{job_id}", response_model=Job)
def get_results(job_id: str):
    """
    Poll this endpoint to get the status and results of a backtest job.
    """
    job = results_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/")
def read_root():
    return {"message": "Backtesting API is running. POST to /run_strategy_3 to start a new backtest job."}
