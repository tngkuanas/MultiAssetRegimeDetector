import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import BacktestRunner from './components/BacktestRunner';
import ResultsDisplay from './components/ResultsDisplay';
import PerformanceChart from './components/PerformanceChart';
import {
  Flex,
  VStack,
  Heading,
  Text,
  ProgressCircle,
  Container,
  Alert,
  AlertTitle,
  AlertDescription,
} from '@chakra-ui/react';
import { Toaster, toaster } from './components/ui/toaster'; // Adjust path if needed
import './App.css'; // Keep custom styles if any

function getDownsampledData(fullData, maxPoints = 500) {
  if (!fullData || fullData.length <= maxPoints) {
    return fullData;
  }

  const step = Math.ceil(fullData.length / maxPoints);
  const downsampled = [];
  for (let i = 0; i < fullData.length; i += step) {
    downsampled.push(fullData[i]);
  }
  return downsampled;
}

function AppContent() {
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState('idle'); // idle, in_progress, completed, error
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const pollResults = useCallback((currentJobId) => {
    let interval;
    interval = setInterval(async () => {
      try {
        const response = await axios.get(`/results/${currentJobId}`);
        if (response.data.status === 'completed') {
          setResults(response.data.result);
          setJobStatus('completed');
          toaster.create({
            title: 'Backtest Completed',
            description: 'Results are ready!',
            status: 'success',
            duration: 5000,
            isClosable: true,
          });
          clearInterval(interval);
        } else if (response.data.status === 'error') {
          const errorMessage = response.data.error || "An unknown error occurred during backtest.";
          setError(errorMessage);
          setJobStatus('error');
          toaster.create({
            title: 'Backtest Error',
            description: errorMessage,
            status: 'error',
            duration: 9000,
            isClosable: true,
          });
          clearInterval(interval);
        }
      } catch (err) {
        console.error("Error polling results:", err);
        const errorMessage = err.response?.data?.detail || "Error fetching results. Please check the backend API.";
        setError(errorMessage);
        setJobStatus('error');
        toaster.create({
          title: 'Error',
          description: errorMessage,
          status: 'error',
          duration: 9000,
          isClosable: true,
        });
        clearInterval(interval);
      }
    }, 3000); // Poll every 3 seconds
    return () => clearInterval(interval); // Cleanup function
  }, []);

  const startBacktest = useCallback(async () => {
    try {
      setJobStatus('in_progress');
      setResults(null);
      setError(null);
      toaster.create({
        title: 'Backtest Started',
        description: 'Initiating backtest, this may take a moment.',
        status: 'info',
        duration: 3000,
        isClosable: true,
      });

      const response = await axios.post('/run_strategy_3');
      const newJobId = response.data.job_id;
      setJobId(newJobId);
    } catch (err) {
      console.error("Error starting backtest:", err);
      const errorMessage = err.response?.data?.detail || "Failed to start backtest. Please check the backend API.";
      setError(errorMessage);
      setJobStatus('error');
      toaster.create({
        title: 'Error',
        description: errorMessage,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  }, []);

  // Effect to clean up interval if component unmounts
  useEffect(() => {
    if (jobStatus === 'in_progress' && jobId) {
      const cleanup = pollResults(jobId);
      return cleanup;
    }
  }, [jobStatus, jobId, pollResults]);


  const downsampledCumulativeReturns = results ? getDownsampledData(results.cumulative_returns) : null;

  return (
    <Container maxW="container.xl" p={4}>
      <Flex direction="column" align="center" justify="center" minH="100vh" py={8}>
        <VStack spacing={8} width="full">
          <Heading as="h1" size="xl" color="teal.500">
            Backtesting Engine - Strategy 3
          </Heading>

          <BacktestRunner onStartBacktest={startBacktest} jobStatus={jobStatus} />

          {jobStatus === 'in_progress' && (
            <Flex direction="column" align="center" mt={4}>
              <ProgressCircle.Root value={null}>
                <ProgressCircle.Circle css={{ "--thickness": "4px" }}>
                  <ProgressCircle.Track />
                  <ProgressCircle.Range stroke="teal.500" />
                </ProgressCircle.Circle>
              </ProgressCircle.Root>
              <Text mt={2}>Backtest in progress... Job ID: {jobId}</Text>
              <Text fontSize="sm" color="gray.500">Please wait, this may take a few minutes.</Text>
            </Flex>
          )}

          {error && (
            <Alert status="error" variant="left-accent" mt={4}>
              <Alert.Indicator />
              <AlertTitle mr={2}>Backtest Failed!</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {results && jobStatus === 'completed' && (
            <VStack spacing={6} width="full" mt={8}>
              <Heading as="h2" size="lg" color="teal.600">
                Results Overview
              </Heading>
              <ResultsDisplay summary={results.summary} />
              {downsampledCumulativeReturns && downsampledCumulativeReturns.length > 0 ? (
                <PerformanceChart data={downsampledCumulativeReturns} />
              ) : (
                <Alert status="warning" mt={4}>
                  <Alert.Indicator />
                  <AlertTitle mr={2}>No Chart Data</AlertTitle>
                  <AlertDescription>Could not generate chart. Data might be insufficient.</AlertDescription>
                </Alert>
              )}
            </VStack>
          )}
        </VStack>
      </Flex>
    </Container>
  );
}

function App() {
  return (
    <>
      <AppContent />
      <Toaster />
    </>
  );
}

export default App;