import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { 
  Box, 
  Flex, 
  VStack, 
  Heading, 
  Text, 
  Spinner,
  Icon
} from '@chakra-ui/react';

import Sidebar from './components/layout/Sidebar';
import MainContent from './components/layout/MainContent';
import Dashboard from './components/dashboard/Dashboard'; 
import { FaCogs } from 'react-icons/fa';

import "react-datepicker/dist/react-datepicker.css";
import "./datepicker-theme.css";
import './App.css';

function App() {
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState('idle'); // idle, in_progress, completed, error
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const pollResults = useCallback((currentJobId) => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get(`http://localhost:8000/results/${currentJobId}`);
        if (response.data.status === 'completed') {
          setResults(response.data.result);
          setJobStatus('completed');
          clearInterval(interval);
        } else if (response.data.status === 'error') {
          setError(response.data.error || "An unknown error occurred.");
          setJobStatus('error');
          clearInterval(interval);
        }
      } catch (err) {
        console.error("Error polling results:", err);
        setError(err.response?.data?.detail || "Error fetching results.");
        setJobStatus('error');
        clearInterval(interval);
      }
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const startBacktest = useCallback(async ({ symbols, startDate, endDate }) => {
    try {
      setJobStatus('in_progress');
      setResults(null);
      setError(null);
      setJobId(null);

      const payload = {
        symbols: symbols.split(',').map(s => s.trim()).filter(Boolean),
        start_date: startDate,
        end_date: endDate,
      };

      const response = await axios.post('http://localhost:8000/run_backtest', payload);
      const newJobId = response.data.job_id;
      setJobId(newJobId);
    } catch (err) {
      console.error("Error starting backtest:", err);
      setError(err.response?.data?.detail || "Failed to start backtest.");
      setJobStatus('error');
    }
  }, []);

  useEffect(() => {
    if (jobStatus === 'in_progress' && jobId) {
      const cleanup = pollResults(jobId);
      return cleanup;
    }
  }, [jobStatus, jobId, pollResults]);

  const renderContent = () => {
    switch (jobStatus) {
      case 'idle':
        return (
          <Flex justify="center" align="center" h="full" opacity={0.3} direction="column">
            <Icon as={FaCogs} w={24} h={24} color="brand.700" />
            <Heading size="lg" mt={6}>READY TO ANALYZE</Heading>
            <Text color="brand.200">Configure portfolio and date range in the sidebar.</Text>
          </Flex>
        );
      case 'in_progress':
        return (
          <Flex justify="center" align="center" h="full" direction="column" bg="rgba(0,0,0,0.5)" backdropFilter="blur(4px)">
            <Spinner thickness="4px" speed="0.65s" color="brand.red" emptyColor="brand.700" size="xl" />
            <Heading size="md" mt={6} letterSpacing="wider">RUNNING SIMULATION...</Heading>
          </Flex>
        );
      case 'error':
        return (
          <Flex justify="center" align="center" h="full" direction="column">
            <Heading size="lg" color="brand.danger">ANALYSIS FAILED</Heading>
            <Text fontFamily="mono" color="brand.200" mt={4} bg="brand.800" p={4} borderRadius="md" maxW="80%">
              {error}
            </Text>
          </Flex>
        );
      case 'completed':
        return results ? <Dashboard results={results} /> : null;
      default:
        return null;
    }
  };

  return (
    <Box position="relative" w="100vw" h="100vh" overflow="hidden" bg="black" sx={{'--grid-pattern-color': 'rgba(255, 255, 255, 0.05)'}}>
      {/* The subtle grid background is applied here */}
      <Box position="absolute" inset="0" className="grid-background" />
      
      <Sidebar onStartBacktest={startBacktest} jobStatus={jobStatus} />
      <MainContent>
        {renderContent()}
      </MainContent>
    </Box>
  );
}

export default App;

