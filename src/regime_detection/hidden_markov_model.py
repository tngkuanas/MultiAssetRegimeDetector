import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
from data_collection import *
from data_preprocessing import *

class HiddenMarkovModel:
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

# def best_number_of_components(data):
#     data = preprocess_data(data)
#     X_train = get_X_train(data).values

#     n_components_range = range(1, 10)
#     scores = []
#     aic = []
#     bic = []

#     for n in n_components_range:
#         gmm = GaussianMixture(
#             n_components=n,
#             covariance_type="full",
#             n_init=50,
#             random_state=7
#         ).fit(X_train)

#         scores.append(gmm.score(X_train))
#         aic.append(gmm.aic(X_train))
#         bic.append(gmm.bic(X_train))

#     for i, n in enumerate(n_components_range):
#         print(f"Components: {n}, Score: {scores[i]:.3f}, AIC: {aic[i]:.1f}, BIC: {bic[i]:.1f}")


# def train_hmm(data):
#     data = preprocess_data(data)
#     X_train = get_X_train(data).values
#     X_train_scaled = StandardScaler().fit_transform(X_train)
#     prices = data["Close"].values.astype(float)

#     hmm_model = GaussianHMM(
#         n_components=4,
#         covariance_type="full",
#         n_iter=100
#     ).fit(X_train_scaled)

#     score = hmm_model.score(X_train_scaled)
#     hidden_states = hmm_model.predict(X_train_scaled)

#     labels_0 = np.where(hidden_states == 0, prices, np.nan)
#     labels_1 = np.where(hidden_states == 1, prices, np.nan)
#     labels_2 = np.where(hidden_states == 2, prices, np.nan)
#     labels_3 = np.where(hidden_states == 3, prices, np.nan)

#     return hidden_states


# def plot_hidden_states(data):
#     hidden_states = train_hmm(data)
#     prices = data["Close"].values.astype(float)

#     print("Correct Number of rows:", len(prices) == len(hidden_states))

#     labels_0, labels_1, labels_2, labels_3 = [], [], [], []

#     for i, s in enumerate(hidden_states):
#         if s == 0:
#             labels_0.append(float(prices[i]))
#             labels_1.append(np.nan)
#             labels_2.append(np.nan)
#             labels_3.append(np.nan)
#         elif s == 1:
#             labels_0.append(np.nan)
#             labels_1.append(float(prices[i]))
#             labels_2.append(np.nan)
#             labels_3.append(np.nan)
#         elif s == 2:
#             labels_0.append(np.nan)
#             labels_1.append(np.nan)
#             labels_2.append(float(prices[i]))
#             labels_3.append(np.nan)
#         elif s == 3:
#             labels_0.append(np.nan)
#             labels_1.append(np.nan)
#             labels_2.append(np.nan)
#             labels_3.append(float(prices[i]))

#     labels_0 = np.array(labels_0)
#     labels_1 = np.array(labels_1)
#     labels_2 = np.array(labels_2)
#     labels_3 = np.array(labels_3)

#     fig = plt.figure(figsize=(18, 10))
#     plt.plot(labels_0, color="red", label="State 0")
#     plt.plot(labels_1, color="green", label="State 1")
#     plt.plot(labels_2, color="black", label="State 2")
#     plt.plot(labels_3, color="orange", label="State 3")
#     plt.legend()
#     plt.show()

#     print(len(labels_0))
