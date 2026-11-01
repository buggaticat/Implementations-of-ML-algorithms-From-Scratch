from __future__ import annotations

from pprint import pformat

import numpy as np


class BaseEstimator:
    """Shared sklearn-like base class for supervised estimators."""

    def __init__(self, **params):
        self.params = dict(params)
        for key, value in params.items():
            setattr(self, key, value)

    def get_params(self, deep=True):
        """Return estimator parameters.

        If ``deep`` is True, nested estimators are expanded using ``__``
        separators, similar to scikit-learn.
        """
        output_params = {}

        for key, value in self.params.items():
            output_params[key] = value

            if deep and hasattr(value, "get_params"):
                nested_params = value.get_params(deep=True)
                for nested_key, nested_value in nested_params.items():
                    output_params[f"{key}__{nested_key}"] = nested_value

        return output_params

    def set_params(self, **params):
        """Set parameters on this estimator and any nested estimators."""
        for key, value in params.items():
            if "__" in key:
                parent_key, child_key = key.split("__", 1)
                if parent_key not in self.params:
                    raise ValueError(f"Unknown parameter '{parent_key}' for {self.__class__.__name__}.")

                nested_estimator = self.params[parent_key]
                if not hasattr(nested_estimator, "set_params"):
                    raise ValueError(
                        f"Parameter '{parent_key}' does not support nested parameter updates."
                    )

                nested_estimator.set_params(**{child_key: value})
            else:
                self.params[key] = value
                setattr(self, key, value)

        return self

    def _check_is_fitted(self, attributes=None):
        """Raise an error if the estimator has not been fitted yet."""
        if attributes is None:
            attributes = [name for name in vars(self) if name.endswith("_")]

        if not attributes:
            raise AttributeError(
                f"{self.__class__.__name__} is not fitted yet. Call 'fit' before using this method."
            )

        missing = [name for name in attributes if not hasattr(self, name)]
        if missing:
            raise AttributeError(
                f"{self.__class__.__name__} is not fitted yet. Missing attributes: {missing}"
            )

    @staticmethod
    def _validate_X(X):
        """Validate and normalize feature input."""
        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must contain numeric values.")
        if not np.isfinite(X).all():
            raise ValueError("X must not contain NaN or infinite values.")

        return X

    @staticmethod
    def _validate_y(y):
        """Validate and normalize target input."""
        y = np.asarray(y)

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        elif y.ndim != 1:
            raise ValueError("y must be a 1D array with one target per sample.")

        if y.shape[0] == 0:
            raise ValueError("y must contain at least one sample.")
        if np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all():
            raise ValueError("y must not contain NaN or infinite values.")

        return y

    @classmethod
    def _validate_X_y(cls, X, y):
        """Validate paired feature and target inputs."""
        X = cls._validate_X(X)
        y = cls._validate_y(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must contain the same number of samples: {X.shape[0]} != {y.shape[0]}"
            )

        return X, y

    def __repr__(self):
        params = ", ".join(
            f"{key}={pformat(value)}" for key, value in self.get_params(deep=False).items()
        )
        return f"{self.__class__.__name__}({params})"
