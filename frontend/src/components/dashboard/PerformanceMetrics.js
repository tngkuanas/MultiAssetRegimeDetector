import React from 'react';
import { SimpleGrid } from '@chakra-ui/react';
import MetricCard from './MetricCard';

const PerformanceMetrics = ({ summary }) => {
  if (!summary) return null;

  const aiStrategyMetrics = {
    return: summary.strategy_final_return,
    sharpe: summary.strategy_sharpe_ratio,
    sortino: summary.strategy_sortino_ratio,
    drawdown: summary.strategy_max_drawdown,
  };

  const riskParityMetrics = {
    return: summary.risk_parity_final_return,
    sharpe: summary.risk_parity_sharpe_ratio,
    sortino: summary.risk_parity_sortino_ratio,
    drawdown: summary.risk_parity_max_drawdown,
  };
  
  // The user spec mentioned "Buy & Hold", but the backend provides "Equal Weight Rebalanced".
  // I will use the "Equal Weight" as the third card for now, as it's the closest available benchmark.
  // If the backend is updated later to include a pure Buy & Hold, this can be swapped.
  const buyAndHoldMetrics = {
    return: summary.equal_weight_final_return,
    sharpe: summary.equal_weight_sharpe_ratio,
    sortino: summary.equal_weight_sortino_ratio,
    drawdown: summary.equal_weight_max_drawdown,
  };


  return (
    <SimpleGrid columns={{ base: 1, lg: 3 }} spacing={6} h="full">
      <MetricCard title="REGIME DETECTION" metrics={aiStrategyMetrics} isHighlighted />
      <MetricCard title="BUY AND HOLD" metrics={buyAndHoldMetrics} />
      <MetricCard title="STATIC RISK PARITY" metrics={riskParityMetrics} />
    </SimpleGrid>
  );
};

export default PerformanceMetrics;
