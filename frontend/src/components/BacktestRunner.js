import React from 'react';
import { Button, Flex, Text } from '@chakra-ui/react';

function BacktestRunner({ onStartBacktest, jobStatus }) {
  const isDisabled = jobStatus === 'in_progress';

  return (
    <Flex direction="column" align="center" mt={4}>
      <Button
        onClick={onStartBacktest}
        isDisabled={isDisabled}
        colorScheme="teal"
        size="lg"
        isLoading={isDisabled}
        loadingText="Running Backtest..."
        spinnerPlacement="start"
      >
        Run Strategy 3 Backtest
      </Button>
      {jobStatus === 'in_progress' && (
        <Text mt={2} color="gray.500">Backtest in progress, please wait...</Text>
      )}
    </Flex>
  );
}

export default BacktestRunner;
