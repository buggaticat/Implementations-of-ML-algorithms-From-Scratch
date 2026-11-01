from __future__ import annotations

import numpy as np


class ClassifierMixin:
    """Shared classification behavior for supervised estimators."""

    _estimator_type = "classifier"

    def score(self, X, y):
        """Return classification accuracy on the given test data and labels."""
        self._check_is_fitted()
        X, y = self._validate_X_y(X, y)

        y_pred = np.asarray(self.predict(X)).ravel()
        y = np.asarray(y).ravel()

        if y_pred.shape[0] != y.shape[0]:
            raise ValueError("predict(X) returned the wrong number of samples.")

        return float(np.mean(y_pred == y))


class RegressorMixin:
    """Shared regression behavior for supervised estimators."""

    _estimator_type = "regressor"

    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction."""
        self._check_is_fitted()
        X, y = self._validate_X_y(X, y)

        y_pred = np.asarray(self.predict(X), dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        if y_pred.shape[0] != y.shape[0]:
            raise ValueError("predict(X) returned the wrong number of samples.")

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return float(1.0 - (ss_res / ss_tot))
