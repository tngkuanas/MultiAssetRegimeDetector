import warnings
import numpy as np
import yaml
from functools import partial
import os

# Suppress warnings from scikit-optimize and other libraries
warnings.filterwarnings("ignore")

# Skopt for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real, Integer

# Project-specific imports
from config_loader import load_config
from portfolio_manager import PortfolioManager
from allocation_strategy import JumpModelRiskParityAllocationStrategy

# --- 1. Define Search Space for Parameters ---
SPACE = [
    Integer(2, 4, name='n_states'),
    Integer(50, 250, name='jump_penalty'),
    Integer(3, 15, name='hysteresis_period'),
    Real(0.1, 0.6, name='max_turnover'),
    Real(0.05, 0.25, name='vol_target_0'),
    Real(0.04, 0.15, name='vol_target_1'),
    Real(0.02, 0.10, name='vol_target_2'),
    Real(0.01, 0.08, name='vol_target_3')
]

# --- 2. Define the Robust Objective Function ---
def objective(params, version_name, portfolio_group, config):
    """
    The objective function for the Bayesian optimizer.
    It now runs the backtest on a GROUP of portfolios.
    Returns a score to be minimized.
    If all portfolios succeed, score is 1.0 - avg_sharpe.
    If not all succeed, score is 1.0 - avg_sharpe + LARGE_PENALTY.
    """
    sjm_params = {
        'n_states': params[0],
        'jump_penalty': params[1],
        'hysteresis_period': params[2],
        'max_turnover': params[3],
        'vol_targets': {i: params[4+i] for i in range(params[0])}
    }
    
    print(f"\nTesting {version_name} with {len(portfolio_group)} portfolios...")
    print(f"Params: n_states={sjm_params['n_states']}, jump_penalty={sjm_params['jump_penalty']}, hyst_period={sjm_params['hysteresis_period']}")

    all_runs_successful_for_params = True # Flag to check if all individual portfolios succeeded
    all_sharpe_scores_for_params = [] # To store Sharpe from all individual runs

    for symbols in portfolio_group:
        current_portfolio_sharpe = 0.0 # Default if run fails
        try:
            manager = PortfolioManager(
                symbols=symbols,
                start_date=config["start_date"],
                end_date=config["end_date"],
                allocation_strategy=JumpModelRiskParityAllocationStrategy(**sjm_params),
                fred_series_to_fetch=config["fred_series"]
            )
            returns, _ = manager.run_portfolio_backtest(plot=False)

            if returns.empty:
                print(f"  - Portfolio [{', '.join(symbols)}]: FAILED (No data).")
                all_runs_successful_for_params = False
                current_portfolio_sharpe = -10.0 # Assign very low Sharpe for failure
            else:
                final_returns = returns[[c for c in returns.columns if 'Cumulative' in c]].iloc[-1]
                strat_ret = final_returns['Strategy Cumulative']
                rp_ret = final_returns['Risk Parity Cumulative']
                ew_ret = final_returns['Equal Weight Rebalanced Cumulative']

                strat_sharpe = manager.sharpe_ratio(returns['Strategy Daily'])
                rp_sharpe = manager.sharpe_ratio(returns['Risk Parity Daily'])
                ew_sharpe = manager.sharpe_ratio(returns['Equal Weight Rebalanced Daily'])
                
                strat_sortino = manager.sortino_ratio(returns['Strategy Daily'])
                rp_sortino = manager.sortino_ratio(returns['Risk Parity Daily'])
                ew_sortino = manager.sortino_ratio(returns['Equal Weight Rebalanced Daily'])
                
                beats_rp = (strat_ret > rp_ret) and (strat_sharpe > rp_sharpe) and (strat_sortino > rp_sortino)
                beats_ew = (strat_ret > ew_ret) and (strat_sharpe > ew_sharpe) and (strat_sortino > ew_sortino)

                if beats_rp and beats_ew:
                    print(f"  - Portfolio [{', '.join(symbols)}]: SUCCESS (Sharpe: {strat_sharpe:.2f})")
                    current_portfolio_sharpe = strat_sharpe
                else:
                    print(f"  - Portfolio [{', '.join(symbols)}]: FAILED (Did not beat benchmarks).")
                    all_runs_successful_for_params = False
                    current_portfolio_sharpe = strat_sharpe # Still record Sharpe, even if it failed criteria

        except Exception as e:
            print(f"  - Portfolio [{', '.join(symbols)}]: EXCEPTION ({e}).")
            all_runs_successful_for_params = False
            current_portfolio_sharpe = -10.0 # Assign very low Sharpe for exception

        all_sharpe_scores_for_params.append(current_portfolio_sharpe)

    # Calculate average Sharpe across all individual backtests in this group
    avg_sharpe_across_group = np.mean(all_sharpe_scores_for_params)

    if all_runs_successful_for_params:
        # If all individual portfolios succeeded, this is a full success. Minimize 1 - avg_sharpe.
        score = 1.0 - avg_sharpe_across_group
        print(f"Result: OVERALL SUCCESS. Avg Sharpe: {avg_sharpe_across_group:.2f}. Score: {score:.3f}")
        return score
    else:
        # If not all individual portfolios succeeded, this is a failure/partial success.
        # Return a score with a penalty. The penalty (e.g., 5.0) ensures that any
        # full success will always have a lower score than any failure.
        score = 1.0 - avg_sharpe_across_group + 5.0
        print(f"Result: PARTIAL/FAIL. Avg Sharpe: {avg_sharpe_across_group:.2f}. Score: {score:.3f}")
        return score


# --- 3. Main Execution Block ---
if __name__ == '__main__':
    print("--- Starting Robust Automated Parameter Optimization ---")
    
    config = load_config()

    # Group of representative portfolios for each version.
    # The 'default_balanced' version is intentionally left out to remain unchanged.
    portfolios_to_tune = {
        "default_balanced": [
            ["VOO", "VXUS", "GLD", "BND", "GOOG"],
            ["SPY", "VEA", "GLD", "AGG", "MSFT"]
        ],
        "broad_etf_equity": [
            ["VOO", "VXUS", "GLD", "BND", "QQQ"], 
            ["VOO", "VXUS", "GLD", "BND", "XLK"]
        ],
        "single_stock_growth": [
            ["VOO", "VXUS", "GLD", "BND", "MSFT"],
            ["VOO", "VXUS", "GLD", "BND", "META"]
        ],
        "bond_heavy": [
            ["VOO", "GLD", "BND", "TLT", "IEF"],
            ["VOO", "VXUS", "BND", "GOVT", "TLT"]
        ]
    }
    
    best_params_found = {}

    for version_name, portfolio_group in portfolios_to_tune.items():
        print(f"\n{'='*20} Optimizing: {version_name} {'='*20}")
        
        objective_partial = partial(objective, version_name=version_name, portfolio_group=portfolio_group, config=config)
        
        result = gp_minimize(
            func=objective_partial,
            dimensions=SPACE,
            n_calls=10,  # Number of different parameter sets to try
            n_initial_points=5, # Number of random points to try before intelligent search
            random_state=42,
            acq_func='EI'
        )
        
        best_score_from_optimizer = result.fun
        best_params_raw = result.x
        
        n_states_val = best_params_raw[0]
        winning_params = {
            'n_states': int(n_states_val),
            'jump_penalty': int(best_params_raw[1]),
            'hysteresis_period': int(best_params_raw[2]),
            'max_turnover': round(best_params_raw[3], 3),
            'vol_targets': {i: round(best_params_raw[4+i], 4) for i in range(n_states_val)}
        }
        
        # Determine if it was a full success or best-effort
        if best_score_from_optimizer < 1.0: # Full success: avg_sharpe > 0 and all beats benchmarks
            print(f"\nSUCCESS for {version_name}! Best Avg Sharpe: {1.0-best_score_from_optimizer:.2f}")
            print(f"Found optimal parameters: {winning_params}")
            best_params_found[version_name] = winning_params
        elif best_score_from_optimizer < 5.0: # Partial success: means it got a non-penalty score
            print(f"\nPARTIAL SUCCESS for {version_name}: No full success, but found best effort parameters.")
            # Unwind the penalty to show actual Avg Sharpe
            actual_avg_sharpe = 1.0 - (best_score_from_optimizer - 5.0)
            print(f"Best Avg Sharpe (best effort): {actual_avg_sharpe:.2f}")
            print(f"Found best effort parameters: {winning_params}")
            best_params_found[version_name] = winning_params
        else: # Likely a total failure, no useful parameters found
            print(f"\nFAILURE for {version_name}: Could not find any useful parameters.")
            # Do not add to best_params_found, so config.yaml remains unchanged for this version.

    # --- 4. Update config.yaml ---
    if best_params_found:
        print(f"\n{'='*20} Updating config.yaml with Winners {'='*20}")
        
        for version_name, params in best_params_found.items():
            if version_name in config['strategy_versions']:
                print(f"Updating parameters for '{version_name}'...")
                config['strategy_versions'][version_name] = params
        
        try:
            with open('config.yaml', 'w') as f:
                yaml.dump(config, f, sort_keys=False, indent=2, width=1000)
            print("Successfully updated config.yaml.")
        except Exception as e:
            print(f"Error writing to config.yaml: {e}")
    else:
        print("\nNo new optimal parameters were found. config.yaml remains unchanged.")

    print("\n--- Optimization Process Complete ---")

    # --- 5. Self-destruct the optimizer script ---
    try:
        print("Removing optimizer script...")
        os.remove(__file__)
    except Exception as e:
        print(f"Could not remove optimizer script: {e}")