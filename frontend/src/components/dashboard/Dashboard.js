import React from 'react';
import { Grid, GridItem } from '@chakra-ui/react';
import { motion } from 'framer-motion';

import PerformanceMetrics from './PerformanceMetrics';
import AlphaChart from './AlphaChart';
import RegimeMap from './RegimeMap';
import AllocationPanel from './AllocationPanel';

const MotionGridItem = motion(GridItem);

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15, // Animate children one by one
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: 'easeOut'
    },
  },
};

const Dashboard = ({ results }) => {
  if (!results) {
    return null;
  }

  return (
    <Grid
      as={motion.div}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      templateRows="auto 1fr auto"
      templateColumns="repeat(2, 1fr)"
      gap={6}
    >
      <MotionGridItem rowSpan={1} colSpan={2} variants={itemVariants}>
        <PerformanceMetrics summary={results.summary} />
      </MotionGridItem>
      
      <MotionGridItem colSpan={1} rowSpan={1} variants={itemVariants} minH="300px">
        <AlphaChart data={results.cumulative_returns} />
      </MotionGridItem>
      
      <MotionGridItem colSpan={1} rowSpan={1} variants={itemVariants} minH="300px">
        <RegimeMap data={results.regime_map_data} />
      </MotionGridItem>
      
      <MotionGridItem colSpan={2} rowSpan={1} variants={itemVariants} minH="250px">
        <AllocationPanel 
          allocationData={results.allocation_data} 
          currentRegime={results.current_regime}
          regimeConfidence={results.regime_confidence}
        />
      </MotionGridItem>
    </Grid>
  );
};

export default Dashboard;
