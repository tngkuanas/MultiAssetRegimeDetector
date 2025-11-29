import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { Box, Heading, Text, Flex } from '@chakra-ui/react';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <Box
        bg="rgba(9, 9, 11, 0.8)"
        backdropFilter="blur(8px)"
        border="1px solid"
        borderColor="brand.700"
        p={3}
        borderRadius="md"
      >
        <Text color="brand.200" fontSize="sm">{`Date: ${label}`}</Text>
        {payload.map((p, i) => (
          <Text key={i} style={{ color: p.color }} fontSize="sm" fontFamily="mono">
            {`${p.name}: ${p.value.toFixed(2)}`}
          </Text>
        ))}
      </Box>
    );
  }
  return null;
};

// Custom Legend for the Alpha Chart
const AlphaLegend = () => {
  const legendItems = [
    { name: 'Regime Detection Model', color: '#dc2626' },
    { name: 'Buy And Hold', color: '#71717a' },
    { name: 'Static Risk Parity', color: '#a1a1aa' },
  ];

  return (
    <Flex justify="center" mt={2} mb={4} wrap="wrap">
      {legendItems.map((item) => (
        <Flex key={item.name} align="center" mr={4}>
          <Box w="3" h="3" bg={item.color} borderRadius="full" mr={2} />
          <Text fontSize="xs" color="brand.200">{item.name}</Text>
        </Flex>
      ))}
    </Flex>
  );
};


const AlphaChart = ({ data }) => {
  const glassmorphismStyle = {
    bg: 'rgba(24, 24, 27, 0.6)',
    backdropFilter: 'blur(12px)',
    border: '1px solid',
    borderColor: 'brand.700',
    borderRadius: 'lg',
    p: 6,
    h: 'full',
    display: 'flex',
    flexDirection: 'column',
  };

  return (
    <Box sx={glassmorphismStyle}>
      <Box>
        <Heading size="md">EQUITY GRAPH</Heading>
        <Text fontSize="sm" color="brand.200">MODEL AND BENCHMARK</Text>
      </Box>
      <AlphaLegend />
      <Box flex="1" minH="0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
            <CartesianGrid stroke="rgba(113, 113, 122, 0.3)" strokeDasharray="3 3" />
            <YAxis
              orientation="right"
              axisLine={false}
              tickLine={false}
              tick={{ fill: '#71717a', fontSize: 12 }}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              name="Regime Detection Model"
              type="monotone"
              dataKey="Strategy Cumulative"
              stroke="#dc2626"
              strokeWidth={2.5}
              dot={false}
            />
            <Line
              name="Buy And Hold"
              type="monotone"
              dataKey="Equal Weight Rebalanced Cumulative"
              stroke="#71717a"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              dot={false}
            />
            <Line
              name="Static Risk Parity"
              type="monotone"
              dataKey="Risk Parity Cumulative"
              stroke="#a1a1aa"
              strokeWidth={1.5}
              strokeDasharray="2 2"
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Box>
  );
};

export default AlphaChart;
