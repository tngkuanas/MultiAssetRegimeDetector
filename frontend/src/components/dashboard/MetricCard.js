import React from 'react';
import {
  Box,
  Heading,
  Text,
  SimpleGrid,
  useTheme,
  chakra,
  Icon
} from '@chakra-ui/react';
import { FaCircle } from 'react-icons/fa';

const MetricCard = ({ title, metrics, isHighlighted = false }) => {
  const theme = useTheme();

  const glassmorphismStyle = {
    bg: isHighlighted ? 'rgba(9, 9, 11, 0.7)' : 'rgba(24, 24, 27, 0.6)',
    backdropFilter: 'blur(12px)',
    border: '1px solid',
    borderColor: 'brand.700',
    borderRadius: 'lg',
    p: 6,
    h: 'full',
    position: 'relative'
  };

  const PulsingDot = chakra(Icon, {
    baseStyle: {
      color: 'brand.red',
      w: 3,
      h: 3,
      position: 'absolute',
      top: 4,
      right: 4,
      animation: `pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite`,
      '@keyframes pulse': {
        '0%, 100%': {
          opacity: 1,
        },
        '50%': {
          opacity: 0.5,
        },
      },
    },
  });

  return (
    <Box sx={glassmorphismStyle}>
      {isHighlighted && <PulsingDot as={FaCircle} />}
      <Heading size="md" mb={4}>{title}</Heading>
      <SimpleGrid columns={2} spacing={4}>
        <Box>
          <Text fontSize="sm" color="brand.200">Total Return</Text>
          <Text fontFamily="mono" fontWeight="bold" fontSize="xl" color={metrics.return > 0 ? 'brand.success' : 'brand.danger'}>
            {(metrics.return * 100).toFixed(2)}%
          </Text>
        </Box>
        <Box>
          <Text fontSize="sm" color="brand.200">Sharpe Ratio</Text>
          <Text fontFamily="mono" fontWeight="bold" fontSize="xl">
            {metrics.sharpe.toFixed(2)}
          </Text>
        </Box>
        <Box>
          <Text fontSize="sm" color="brand.200">Sortino Ratio</Text>
          <Text fontFamily="mono" fontWeight="bold" fontSize="xl">
            {metrics.sortino.toFixed(2)}
          </Text>
        </Box>
        <Box>
          <Text fontSize="sm" color="brand.200">Max Drawdown</Text>
          <Text fontFamily="mono" fontWeight="bold" fontSize="xl" color="brand.danger">
            {(metrics.drawdown * 100).toFixed(2)}%
          </Text>
        </Box>
      </SimpleGrid>
    </Box>
  );
};

export default MetricCard;
