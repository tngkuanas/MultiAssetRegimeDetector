import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class StatisticalJumpModel:
    """
    Implements a Statistical Jump Model (SJM) for regime detection.
    
    This model clusters time-series data into a specified number of regimes
    using an iterative reassignment algorithm, which is a variation of K-Means.
    It includes a penalty for switching regimes between consecutive time steps
    to promote regime persistence.
    """
    def __init__(self, n_states=3, jump_penalty=100, max_iter=100, random_state=42):
        """
        Initializes the StatisticalJumpModel.

        :param n_states: The number of regimes (clusters) to identify.
        :param jump_penalty: The cost ($\lambda$) for switching regimes between
                             consecutive time steps. Higher values lead to more
                             persistent regimes.
        :param max_iter: The maximum number of iterations for the clustering algorithm.
        :param random_state: Seed for the random number generator for reproducibility.
        """
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.regime_labels_ = None

    def fit(self, features):
        """
        Fits the SJM to the provided feature data using iterative reassignment
        with a Viterbi-like dynamic programming approach to enforce persistence.

        :param features: A DataFrame of standardized features, indexed by date.
        :return: A pandas Series containing the final regime labels, aligned with the feature index.
        """
        X = features.values
        n_samples = X.shape[0]

        # 1. Initial assignment using KMeans to get starting centroids
        kmeans = KMeans(n_clusters=self.n_states, random_state=self.random_state, n_init=10)
        initial_regimes = kmeans.fit_predict(X)
        
        # Initial centroids based on KMeans
        centroids = kmeans.cluster_centers_
        
        # Iterative reassignment loop
        for i in range(self.max_iter):
            # Dynamic Programming step (Viterbi-like path finding)
            cost_matrix = np.zeros((n_samples, self.n_states))
            path_matrix = np.zeros((n_samples, self.n_states), dtype=int)

            # Calculate initial costs for the first time step
            for j in range(self.n_states):
                cost_matrix[0, j] = np.sum((X[0] - centroids[j]) ** 2)
            
            # Fill the cost and path matrices
            for t in range(1, n_samples):
                for j in range(self.n_states):
                    # Cost to be in state j at time t
                    dist_cost = np.sum((X[t] - centroids[j]) ** 2)
                    
                    # Find the minimum cost to transition to state j
                    costs_from_prev = cost_matrix[t-1] + self.jump_penalty * (np.arange(self.n_states) != j)
                    
                    min_prev_state = np.argmin(costs_from_prev)
                    min_cost = costs_from_prev[min_prev_state]
                    
                    cost_matrix[t, j] = dist_cost + min_cost
                    path_matrix[t, j] = min_prev_state

            # Backtrack to find the optimal path
            new_regimes = np.zeros(n_samples, dtype=int)
            new_regimes[-1] = np.argmin(cost_matrix[-1])
            for t in range(n_samples - 2, -1, -1):
                new_regimes[t] = path_matrix[t + 1, new_regimes[t + 1]]

            # Check for convergence
            if i > 0 and np.array_equal(new_regimes, old_regimes):
                break
            
            old_regimes = new_regimes

            # Update centroids for the next iteration
            for j in range(self.n_states):
                # Check if state j is present in the new regimes
                if np.any(new_regimes == j):
                    centroids[j] = X[new_regimes == j].mean(axis=0)
        
        # --- Sort regimes by volatility for interpretability ---
        # The 'volatility' feature is assumed to be the first column
        vol_by_regime = pd.Series(features.iloc[:, 0]).groupby(new_regimes).mean()
        regime_order = vol_by_regime.sort_values().index
        
        # Create a mapping from old regime labels to new, sorted ones (0=low, 1=mid, 2=high)
        remap_dict = {old_label: new_label for new_label, old_label in enumerate(regime_order)}
        
        final_regimes = np.array([remap_dict[r] for r in new_regimes])
        
        # Store sorted centroids and labels
        self.centroids = centroids[[remap_dict.get(i, i) for i in range(self.n_states)]]
        self.regime_labels_ = pd.Series(final_regimes, index=features.index, name='regime')
        
        return self.regime_labels_
