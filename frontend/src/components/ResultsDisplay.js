import React from 'react';
import { Table, Box, Heading } from '@chakra-ui/react';

function ResultsDisplay({ summary }) {
  if (!summary) {
    return null;
  }

  const formatPercentage = (value) => `${(value * 100).toFixed(2)}%`;
  const formatRatio = (value) => value.toFixed(2);

  return (
    <Box>
      <Heading as="h3" size="md" mb={4}>Performance Summary</Heading>
      <Table.Root variant="simple" size="sm">
        <Table.Header>
          <Table.Row>
            <Table.ColumnHeader>Metric</Table.ColumnHeader>
            <Table.ColumnHeader>Strategy</Table.ColumnHeader>
            <Table.ColumnHeader>Risk Parity Benchmark</Table.ColumnHeader>
            <Table.ColumnHeader>Equal Weight Benchmark</Table.ColumnHeader>
          </Table.Row>
        </Table.Header>
        <Table.Body>
          <Table.Row>
            <Table.Cell>Final Return</Table.Cell>
            <Table.Cell>{formatPercentage(summary.strategy_final_return)}</Table.Cell>
            <Table.Cell>{formatPercentage(summary.risk_parity_final_return)}</Table.Cell>
            <Table.Cell>{formatPercentage(summary.equal_weight_final_return)}</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.Cell>Sharpe Ratio</Table.Cell>
            <Table.Cell>{formatRatio(summary.strategy_sharpe_ratio)}</Table.Cell>
            <Table.Cell>{formatRatio(summary.risk_parity_sharpe_ratio)}</Table.Cell>
            <Table.Cell>{formatRatio(summary.equal_weight_sharpe_ratio)}</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.Cell>Sortino Ratio</Table.Cell>
            <Table.Cell>{formatRatio(summary.strategy_sortino_ratio)}</Table.Cell>
            <Table.Cell>{formatRatio(summary.risk_parity_sortino_ratio)}</Table.Cell>
            <Table.Cell>{formatRatio(summary.equal_weight_sortino_ratio)}</Table.Cell>
          </Table.Row>
          <Table.Row>
            <Table.Cell>Max Drawdown</Table.Cell>
            <Table.Cell>{formatPercentage(summary.strategy_max_drawdown)}</Table.Cell>
            <Table.Cell>{formatPercentage(summary.risk_parity_max_drawdown)}</Table.Cell>
            <Table.Cell>{formatPercentage(summary.equal_weight_max_drawdown)}</Table.Cell>
          </Table.Row>
        </Table.Body>
      </Table.Root>
    </Box>
  );
}

export default ResultsDisplay;
