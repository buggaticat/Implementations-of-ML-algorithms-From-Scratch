import numpy as np

from _linear_regression import LinearRegression

class LassoRegression(LinearRegression):
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        super().__init__(fit_intercept=fit_intercept)

        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X] 
        
        n, p = X.shape
        beta = np.zeros(p)
        residual = y - X @ beta

        for iteration in range(self.max_iter):
            old_beta = beta.copy()

            for j in range(p):
                X_j = X[:, j]
                residual += (X_j * beta[j])
                rho = (X_j @ residual) / n
                z = (X_j.T @ X_j) / n

                penalty = 0 if self.fit_intercept and j == 0 else self.alpha

                if z == 0:
                    beta[j] = 0.0
                else:
                    beta[j] = (np.sign(rho) * max(abs(rho) - penalty, 0.0)) / z

                residual -= X[:, j] * beta[j]
            
            if np.max(np.abs(beta - old_beta)) < self.tol:
                break

        if self.fit_intercept:
            self.coef_ = beta[1:]
            self.intercept_ = beta[0]
        else:
            self.coef_ = beta
            self.intercept_ = 0.0

        self.n_iter_ = iteration + 1
        
        return self