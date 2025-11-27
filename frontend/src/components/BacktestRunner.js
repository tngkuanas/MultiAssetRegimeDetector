import React from 'react';

function BacktestRunner({ onStartBacktest, jobStatus }) {
  const isDisabled = jobStatus === 'in_progress';

  return (
    <div className="backtest-runner">
      <button onClick={onStartBacktest} disabled={isDisabled}>
        {isDisabled ? 'Running Backtest...' : 'Run Strategy 3 Backtest'}
      </button>
      {jobStatus === 'in_progress' && (
        <div className="spinner"></div> // Simple spinner for visual feedback
      )}
    </div>
  );
}

export default BacktestRunner;
