import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from data_preprocessing import preprocess_data

class HiddenMarkovModel:
    def __init__(self, n_components=4, train_window_size=500):
        self.n_components = n_components
        self.train_window_size = train_window_size
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=42,
            init_params="",
        )

    def generate_psignal(self, asset_data, macro_data=None):
        """
        Generates the PSignal using a walk-forward training approach for the HMM.
        """
        # 1. Preprocess data
        preprocessed_data = preprocess_data(asset_data)
        if macro_data is not None:
            preprocessed_data = preprocessed_data.join(macro_data)
        preprocessed_data.ffill(inplace=True)
        preprocessed_data.dropna(inplace=True)

        hmm_features = ["Returns", "Range"]
        if macro_data is not None:
            hmm_features.extend(macro_data.columns)
        
        hmm_features = [f for f in hmm_features if f in preprocessed_data.columns]
        
        scaler = StandardScaler()
        all_psignals = pd.Series(index=preprocessed_data.index)

        # 2. Walk-forward training and prediction
        for i in range(self.train_window_size, len(preprocessed_data)):
            train_data = preprocessed_data.iloc[:i]
            predict_data = preprocessed_data.iloc[i : i + 1]

            if predict_data.empty:
                break

            # Scale data for the current window
            X_train_scaled = scaler.fit_transform(train_data[hmm_features])
            
            # Fit the model
            self.model.fit(X_train_scaled)

            # Determine favourable states on the training data
            train_preds = self.model.predict(X_train_scaled)
            train_with_states = train_data.copy()
            train_with_states["State"] = train_preds
            state_returns = train_with_states.groupby("State")["Returns"].mean()
            favourable_states = state_returns[state_returns > 0].index.tolist()

            # Predict the state for the next day
            X_predict_scaled = scaler.transform(predict_data[hmm_features])
            next_day_pred = self.model.predict(X_predict_scaled)[0]
            
            # Assign PSignal
            all_psignals.loc[predict_data.index[0]] = (
                1 if next_day_pred in favourable_states else 0
            )
            
        return all_psignals.fillna(0)
