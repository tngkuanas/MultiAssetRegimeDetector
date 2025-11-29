import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceArea,
  Legend,
} from 'recharts';
import { Box, Heading, Text, Flex } from '@chakra-ui/react';

// This is the same custom tooltip from AlphaChart. It could be moved to a shared file.
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
          <Text style={{ color: '#dc2626' }} fontSize="sm" fontFamily="mono">
            {`Value: ${payload[0].value.toFixed(2)}`}
          </Text>
        </Box>
      );
    }
    return null;
  };

// Helper function to group consecutive regimes for drawing ReferenceAreas
const getRegimeAreas = (data) => {
    if (!data || data.length === 0) {
      return [];
    }
  
    const areas = [];
    let currentArea = {
      regime: data[0].regime,
      x1: data[0].Date,
      x2: data[0].Date,
    };
  
    for (let i = 1; i < data.length; i++) {
      if (data[i].regime === currentArea.regime) {
        currentArea.x2 = data[i].Date;
      } else {
        areas.push(currentArea);
        currentArea = {
          regime: data[i].regime,
          x1: data[i].Date,
          x2: data[i].Date,
        };
      }
    }
    areas.push(currentArea); // Add the last area
  
    return areas;
};

// Custom Legend for Regimes
const RegimeLegend = () => {
  const regimeColorMap = {
    0: 'rgba(16, 185, 129, 0.25)', // Low Vol -> Bullish -> Green
    1: 'rgba(113, 113, 122, 0.25)', // Mid Vol -> Neutral -> Gray
    2: 'rgba(239, 68, 68, 0.25)', // High Vol -> Bearish -> Red
  };
  const regimeNames = {
    0: 'BULLISH QUIET',
    1: 'NEUTRAL',
    2: 'BEARISH VOLATILE',
  };

  return (
    <Flex justify="center" mt={2} mb={4} wrap="wrap">
      {Object.entries(regimeNames).map(([key, name]) => (
        <Flex key={key} align="center" mr={4}>
          <Box w="3" h="3" bg={regimeColorMap[parseInt(key)]} borderRadius="full" mr={1} />
          <Text fontSize="xs" color="brand.200" textTransform="uppercase">{name}</Text>
        </Flex>
      ))}
    </Flex>
  );
};

const RegimeMap = ({ data }) => {
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

    const regimeAreas = getRegimeAreas(data);
    const regimeColorMap = {
      0: 'rgba(16, 185, 129, 0.25)', // Low Vol -> Bullish -> Green
      1: 'rgba(113, 113, 122, 0.25)', // Mid Vol -> Neutral -> Gray
      2: 'rgba(239, 68, 68, 0.25)', // High Vol -> Bearish -> Red
    };

    return (
        <Box sx={glassmorphismStyle}>
            <Box>
                <Heading size="md">REGIME DETECTION MAP</Heading>
                <Text fontSize="sm" color="brand.200">VOLATILITY SHADING</Text>
            </Box>
            <RegimeLegend /> {/* Custom legend for regimes */}
            <Box flex="1" minH="0"> {/* Wrapper to allow chart to shrink */}
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                        <CartesianGrid stroke="rgba(113, 113, 122, 0.3)" strokeDasharray="3 3" />
                        <XAxis dataKey="Date" hide />
                        <YAxis
                            orientation="right"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: '#71717a', fontSize: 12 }}
                            tickFormatter={(value) => value.toFixed(2)}
                            yAxisId={0}
                        />
                        <Tooltip content={<CustomTooltip />} />

                        {/* Render the background shades for regimes */}
                        {regimeAreas.map((area, index) => (
                            <ReferenceArea
                                key={index}
                                x1={area.x1}
                                x2={area.x2}
                                yAxisId={0}
                                stroke="none"
                                fill={regimeColorMap[area.regime]}
                                ifOverflow="visible"
                            />
                        ))}

                        <Line
                            name="Regime Detection Model"
                            type="monotone"
                            dataKey="value"
                            stroke="#dc2626"
                            strokeWidth={2}
                            dot={false}
                        />
                    </LineChart>
                </ResponsiveContainer>
            </Box>
        </Box>
    );
};

export default RegimeMap;
