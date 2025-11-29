from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from typing import Dict, Any, Optional, List
import uuid
import time

# Import your existing backtesting logic
from config_loader import load_config
from portfolio_manager import PortfolioManager
from allocation_strategy import JumpModelRiskParityAllocationStrategy
from strategy_selector import get_portfolio_composition, select_strategy_version

app = FastAPI()

# --- CORS Middleware ---
# Allows the frontend (running on localhost:3000) to communicate with the backend.
origins = [
    "http://localhost:3000",
    "http://localhost",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- In-Memory Storage for Job Results ---
results_store: Dict[str, Dict[str, Any]] = {}


# --- Pydantic Models for API ---

class BacktestRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str

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

class DashboardData(BaseModel):
    summary: BacktestSummary
    cumulative_returns: list[Dict[str, Any]]
    regime_map_data: list[Dict[str, Any]]
    allocation_data: list[Dict[str, Any]]
    current_regime: str
    regime_confidence: float

class Job(BaseModel):
    job_id: str
    status: str
    result: Optional[DashboardData] = None
    error: Optional[str] = None


# --- Helper Function to run the backtest ---

def run_backtest_and_store_results(job_id: str, request: BacktestRequest):
    """
    A wrapper function that runs the backtest and stores the result.
    This function is designed to be run in the background.
    """
    try:
        print(f"Starting backtest for job_id: {job_id}")
        config = load_config()
        # Use request data instead of config for primary backtest params
        symbols = request.symbols
        start_date = request.start_date
        end_date = request.end_date
        fred_series = config["fred_series"]
        
        # --- Strategy Selection ---
        asset_definitions = config.get("asset_definitions", {})
        selection_rules = config.get("strategy_selection_rules", [])
        
        composition = get_portfolio_composition(symbols, asset_definitions)
        version_name = select_strategy_version(composition, selection_rules)
        
        print(f"Selected strategy version: '{version_name}' for symbols: {', '.join(symbols)}")
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
        
        # NOTE: This will require changes in portfolio_manager to return regime_labels
        portfolio_returns, weights_df, regime_labels = portfolio_manager.run_portfolio_backtest(plot=False)

        # --- Format the results for the new dashboard ---
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

        # Format cumulative returns for chart
        cumulative_returns_chart_data = portfolio_returns[['Strategy Cumulative', 'Risk Parity Cumulative', 'Equal Weight Rebalanced Cumulative']].reset_index()
        cumulative_returns_chart_data['Date'] = cumulative_returns_chart_data['Date'].dt.strftime('%Y-%m-%d')
        
        # Format regime data for regime map
        regime_map_data_df = pd.DataFrame({
            'Date': regime_labels.index,
            'regime': regime_labels.values,
            'value': portfolio_returns.loc[regime_labels.index]['Strategy Cumulative']
        })
        regime_map_data_df['Date'] = regime_map_data_df['Date'].dt.strftime('%Y-%m-%d')

        # Format allocation data
        final_weights = weights_df.iloc[-1]
        allocation_data = [{'asset': symbol, 'weight': weight * 100} for symbol, weight in final_weights.items()]

        # Placeholder for current regime text and confidence
        regime_map = {0: "BULLISH QUIET", 1: "NEUTRAL", 2: "BEARISH VOLATILE"}
        current_regime_code = int(regime_labels.iloc[-1])
        current_regime_text = regime_map.get(current_regime_code, "UNKNOWN")
        
        result_data = DashboardData(
            summary=summary,
            cumulative_returns=cumulative_returns_chart_data.to_dict(orient='records'),
            regime_map_data=regime_map_data_df.to_dict(orient='records'),
            allocation_data=allocation_data,
            current_regime=current_regime_text,
            regime_confidence=0.95 # Placeholder value
        )
        
        results_store[job_id].update({"status": "completed", "result": result_data})
        print(f"Completed backtest for job_id: {job_id}")

    except Exception as e:
        print(f"Error running backtest for job_id: {job_id}. Error: {e}")
        results_store[job_id].update({"status": "error", "error": str(e)})


# --- API Endpoints ---

@app.post("/run_backtest", response_model=Job)
def run_backtest_endpoint(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Kicks off the backtest in the background.
    """
    job_id = str(uuid.uuid4())
    results_store[job_id] = {
        "job_id": job_id, 
        "status": "in_progress", 
        "result": None, 
        "error": None
    }
    background_tasks.add_task(run_backtest_and_store_results, job_id, request)
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
