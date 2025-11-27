import React, { useState } from 'react';
import axios from 'axios';
import BacktestRunner from './components/BacktestRunner';
import ResultsDisplay from './components/ResultsDisplay';
import PerformanceChart from './components/PerformanceChart';
import './App.css'; // Import the CSS file

function App() {
  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState('idle'); // idle, in_progress, completed, error
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const startBacktest = async () => {
    try {
      setJobStatus('in_progress');
      setResults(null);
      setError(null);
      const response = await axios.post('/run_strategy_3');
      setJobId(response.data.job_id);
      pollResults(response.data.job_id);
    } catch (err) {
      console.error("Error starting backtest:", err);
      setError("Failed to start backtest. Please check the backend.");
      setJobStatus('error');
    }
  };

  const pollResults = async (currentJobId) => {
    let interval;
    interval = setInterval(async () => {
      try {
        const response = await axios.get(`/results/${currentJobId}`);
        if (response.data.status === 'completed') {
          setResults(response.data.result);
          setJobStatus('completed');
          clearInterval(interval);
        } else if (response.data.status === 'error') {
          setError(response.data.error || "An unknown error occurred during backtest.");
          setJobStatus('error');
          clearInterval(interval);
        }
      } catch (err) {
        console.error("Error polling results:", err);
        setError("Error fetching results. Please check the backend.");
        setJobStatus('error');
        clearInterval(interval);
      }
    }, 3000); // Poll every 3 seconds
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Backtesting Engine - Strategy 3</h1>
      </header>
      <main className="App-main">
        <BacktestRunner onStartBacktest={startBacktest} jobStatus={jobStatus} />

        {jobStatus === 'in_progress' && <p>Backtest in progress... Job ID: {jobId}</p>}
        {error && <p className="error-message">Error: {error}</p>}

        {results && jobStatus === 'completed' && (
          <div className="results-container">
            <h2>Backtest Results</h2>
            <ResultsDisplay summary={results.summary} />
            <PerformanceChart data={results.cumulative_returns} />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;