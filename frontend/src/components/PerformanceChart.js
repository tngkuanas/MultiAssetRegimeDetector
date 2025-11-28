import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Heading, Box } from '@chakra-ui/react';

const PerformanceChart = React.memo(({ data }) => {
  if (!data || data.length === 0) {
    return null;
  }

  const processedData = data.map(item => ({
    Date: item.Date,
    Strategy: item['Strategy Cumulative'],
    RiskParity: item['Risk Parity Cumulative'],
    EqualWeight: item['Equal Weight Rebalanced Cumulative']
  }));

  return (
    <Box className="performance-chart" width="100%">
      <Heading as="h3" size="md" mb={4}>Cumulative Returns</Heading>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={processedData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="Date" />
          <YAxis tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
          <Tooltip formatter={(value) => `${(value * 100).toFixed(2)}%`} />
          <Legend />
          <Line type="monotone" dataKey="Strategy" stroke="#8884d8" activeDot={{ r: 8 }} />
          <Line type="monotone" dataKey="RiskParity" stroke="#82ca9d" />
          <Line type="monotone" dataKey="EqualWeight" stroke="#ffc658" />
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
});

export default PerformanceChart;
