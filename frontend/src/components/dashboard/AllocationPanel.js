import React from 'react';
import {
  Box,
  Heading,
  Text,
  VStack,
  Flex,
  Progress,
} from '@chakra-ui/react';
import { motion } from 'framer-motion';

const MotionBox = motion(Box);

const AllocationRow = ({ rank, asset, weight }) => {
  const rankStr = String(rank).padStart(2, '0');

  return (
    <Flex align="center" justify="space-between" w="full">
      <Flex align="center" flex="1" minW={0} mr={4}>
        <Text color="brand.200" fontFamily="mono" fontSize="sm" mr={4}>{rankStr}</Text>
        <Text fontWeight="bold" isTruncated>{asset}</Text>
      </Flex>
      <Flex align="center" w="180px">
        <Progress
          value={weight}
          w="full"
          colorScheme="red"
          bg="brand.800"
          size="sm"
          borderRadius="sm"
          mr={3}
        />
        <Text fontFamily="mono" fontSize="sm">{weight.toFixed(2)}%</Text>
      </Flex>
    </Flex>
  );
};


const AllocationPanel = ({ allocationData, currentRegime, regimeConfidence }) => {
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

  const regimeColor = currentRegime.includes('BULLISH') ? 'brand.success' :
                      currentRegime.includes('BEARISH') ? 'brand.danger' :
                      'brand.200';

  // Sort allocation by weight descending
  const sortedAllocation = [...allocationData].sort((a, b) => b.weight - a.weight);

  return (
    <Box sx={glassmorphismStyle}>
      <Box mb={4}>
        <Heading size="md">REGIME & ALLOCATION</Heading>
        <Text fontSize="sm" color="brand.200">AI MODEL V2</Text>
      </Box>

      {/* Section A: Regime Indicator */}
      <Flex
        border="1px solid"
        borderColor={regimeColor}
        borderRadius="md"
        p={4}
        justify="space-between"
        align="center"
        mb={4}
      >
        <Heading size="lg" color={regimeColor}>{currentRegime}</Heading>
        <Box textAlign="right">
            <Text fontSize="xs" color="brand.200">CONFIDENCE</Text>
            <Text fontFamily="mono" fontSize="2xl" fontWeight="bold">
                {(regimeConfidence * 100).toFixed(0)}%
            </Text>
        </Box>
      </Flex>

      {/* Section B: Allocation List */}
      <VStack spacing={3} align="stretch" mb={4}>
        {sortedAllocation.map((item, index) => (
          <AllocationRow
            key={item.asset}
            rank={index + 1}
            asset={item.asset}
            weight={item.weight}
          />
        ))}
      </VStack>
    </Box>
  );
};

export default AllocationPanel;
