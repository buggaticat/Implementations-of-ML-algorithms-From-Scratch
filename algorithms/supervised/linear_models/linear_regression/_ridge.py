import numpy as np

from _linear_regression import LinearRegression

class RidgeRegression(LinearRegression):
    def __init__(self, alpha=1.0, fit_intercept=True):
        super().__init__(fit_intercept=fit_intercept)
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        I = np.eye(X.shape[1])

        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X] 
            I[0, 0] = 0
        
        A = (X.T @ X) + (self.alpha * I)
        B = X.T @ y
        beta = np.linalg.solve(A, B)
        if self.fit_intercept:
            self.coef_ = beta[1:]
            self.intercept_ = beta[0]
        else:
            self.coef_ = beta
            self.intercept_ = 0.0
        
        return self