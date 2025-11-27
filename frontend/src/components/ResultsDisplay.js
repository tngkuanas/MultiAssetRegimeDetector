import React from 'react';

function ResultsDisplay({ summary }) {
  if (!summary) {
    return null;
  }

  const formatPercentage = (value) => `${(value * 100).toFixed(2)}%`;
  const formatRatio = (value) => value.toFixed(2);

  return (
    <div className="results-display">
      <h3>Performance Summary</h3>
      <table className="summary-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>Strategy</th>
            <th>Risk Parity Benchmark</th>
            <th>Equal Weight Benchmark</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Final Return</td>
            <td>{formatPercentage(summary.strategy_final_return)}</td>
            <td>{formatPercentage(summary.risk_parity_final_return)}</td>
            <td>{formatPercentage(summary.equal_weight_final_return)}</td>
          </tr>
          <tr>
            <td>Sharpe Ratio</td>
            <td>{formatRatio(summary.strategy_sharpe_ratio)}</td>
            <td>{formatRatio(summary.risk_parity_sharpe_ratio)}</td>
            <td>{formatRatio(summary.equal_weight_sharpe_ratio)}</td>
          </tr>
          <tr>
            <td>Sortino Ratio</td>
            <td>{formatRatio(summary.strategy_sortino_ratio)}</td>
            <td>{formatRatio(summary.risk_parity_sortino_ratio)}</td>
            <td>{formatRatio(summary.equal_weight_sortino_ratio)}</td>
          </tr>
          <tr>
            <td>Max Drawdown</td>
            <td>{formatPercentage(summary.strategy_max_drawdown)}</td>
            <td>{formatPercentage(summary.risk_parity_max_drawdown)}</td>
            <td>{formatPercentage(summary.equal_weight_max_drawdown)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export default ResultsDisplay;
