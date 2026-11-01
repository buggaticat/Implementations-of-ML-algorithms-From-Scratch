import numpy as np
from ...base import BaseEstimator
from ...mixins import RegressorMixin

class LinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_features_in_ = None
        self.rank_ = None
        self.singular_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
        beta = Vh.T @ (np.linalg.pinv(np.diag(S))) @ U.T @ y

        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.coef_ = beta
            self.intercept_ = 0.0

        self.singular_ = S
        self.rank_ = np.linalg.matrix_rank(X)

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
        
            