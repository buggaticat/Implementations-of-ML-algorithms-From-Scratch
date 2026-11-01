# Supervised Learning Library Plan

## Summary
This folder will contain the supervised-learning portion of the library, designed with a scikit-learn-like API. Each algorithm should be represented as a class with a consistent interface, so users can instantiate a model, call `fit`, and then use prediction-related methods in a predictable way.

The goal is not just to implement algorithms, but to make them feel like a structured library that is easy to learn, easy to extend, and easy to use.

## API Design Principles
All supervised estimators should follow the same general pattern:

- Constructor takes only hyperparameters
- `fit(...)` trains the model and returns `self`
- `predict(...)` returns outputs for new inputs
- Classifiers may expose `predict_proba(...)` and `decision_function(...)` when appropriate
- Regressors expose `score(...)` using regression-appropriate scoring
- Classifiers expose `score(...)` using accuracy-style scoring
- Learned attributes use trailing underscores, for example:
  - `coef_`
  - `intercept_`
  - `classes_`
  - `tree_`
  - `estimators_`

A shared base layer should be used for consistent behavior across all estimators.

## Planned Algorithms

### Linear Models
These models form the foundation of supervised learning and should be implemented first.

#### `LinearRegression`
Methods:
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`

Purpose:
- Basic continuous-value prediction
- Baseline regression model

#### `LogisticRegression`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `decision_function(X)`
- `score(X, y)`

Purpose:
- Binary classification
- Should support class probabilities and decision values

#### `SoftmaxRegression`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `decision_function(X)`
- `score(X, y)`

Purpose:
- Multiclass classification
- Softmax-based generalization of logistic regression

---

### Naive Bayes
These should be simple, probabilistic classifiers with a clean API.

#### `GaussianNB`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `predict_log_proba(X)`
- `score(X, y)`

Purpose:
- Continuous-valued feature classification with Gaussian assumption

#### `MultinomialNB`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `predict_log_proba(X)`
- `score(X, y)`

Purpose:
- Count-based features
- Common for text-like or frequency data

#### `BernoulliNB`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `predict_log_proba(X)`
- `score(X, y)`

Purpose:
- Binary-feature classification

---

### K-Nearest Neighbors
These should provide simple instance-based learning behavior.

#### `KNeighborsClassifier`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `kneighbors(X)`
- `score(X, y)`

Purpose:
- Classification by nearest samples

#### `KNeighborsRegressor`
Methods:
- `fit(X, y)`
- `predict(X)`
- `kneighbors(X)`
- `score(X, y)`

Purpose:
- Regression by nearest samples

---

### Support Vector Machines
These should support both linear and kernel-based variants where relevant.

#### `SVC`
Methods:
- `fit(X, y)`
- `predict(X)`
- `decision_function(X)`
- `predict_proba(X)` when enabled
- `score(X, y)`

Purpose:
- Kernel-based classification

#### `SVR`
Methods:
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`

Purpose:
- Kernel-based regression

#### `LinearSVC`
Methods:
- `fit(X, y)`
- `predict(X)`
- `decision_function(X)`
- `score(X, y)`

Purpose:
- Linear classification at scale

#### `LinearSVR`
Methods:
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`

Purpose:
- Linear regression at scale

---

### Decision Trees
These should form the core tree-based building block for ensembles later.

#### `DecisionTreeClassifier`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `apply(X)`
- `decision_path(X)`
- `get_depth()`
- `get_n_leaves()`
- `score(X, y)`

Purpose:
- Interpretable classification tree

#### `DecisionTreeRegressor`
Methods:
- `fit(X, y)`
- `predict(X)`
- `apply(X)`
- `decision_path(X)`
- `get_depth()`
- `get_n_leaves()`
- `score(X, y)`

Purpose:
- Interpretable regression tree

---

### Random Forests
These should wrap multiple decision trees and combine them through bagging-style aggregation.

#### `RandomForestClassifier`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `apply(X)`
- `score(X, y)`

Purpose:
- Ensemble classification using many trees

#### `RandomForestRegressor`
Methods:
- `fit(X, y)`
- `predict(X)`
- `apply(X)`
- `score(X, y)`

Purpose:
- Ensemble regression using many trees

---

### Boosting Family
This section should cover the standard boosting variants in a sklearn-like style.

#### `GradientBoostingClassifier`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `decision_function(X)`
- `staged_predict(X)`
- `staged_predict_proba(X)`
- `score(X, y)`

Purpose:
- Additive tree boosting for classification

#### `GradientBoostingRegressor`
Methods:
- `fit(X, y)`
- `predict(X)`
- `staged_predict(X)`
- `score(X, y)`

Purpose:
- Additive tree boosting for regression

#### `AdaBoostClassifier`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)` when supported
- `score(X, y)`

Purpose:
- Adaptive boosting for classification

#### `AdaBoostRegressor`
Methods:
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`

Purpose:
- Adaptive boosting for regression

#### `XGBoostClassifier`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `score(X, y)`

Purpose:
- Gradient-boosted tree classification with modern optimization behavior

#### `XGBoostRegressor`
Methods:
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`

Purpose:
- Gradient-boosted tree regression

#### `LightGBMClassifier`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `score(X, y)`

Purpose:
- Histogram-based gradient boosting classification

#### `LightGBMRegressor`
Methods:
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`

Purpose:
- Histogram-based gradient boosting regression

#### `CatBoostClassifier`
Methods:
- `fit(X, y)`
- `predict(X)`
- `predict_proba(X)`
- `score(X, y)`

Purpose:
- Categorical-aware gradient boosting classification

#### `CatBoostRegressor`
Methods:
- `fit(X, y)`
- `predict(X)`
- `score(X, y)`

Purpose:
- Categorical-aware gradient boosting regression

## Suggested Package Structure
The codebase should be organized by algorithm family.

Recommended layout:
- `supervised/base.py`
- `supervised/linear_models/`
- `supervised/naive_bayes/`
- `supervised/neighbors/`
- `supervised/svm/`
- `supervised/trees/`
- `supervised/ensembles/forest/`
- `supervised/ensembles/boosting/`

## Testing Expectations
Every estimator should have tests that verify the same core behaviors:

- Constructor and parameter handling
- `fit` / `predict` flow
- `score` behavior
- probability outputs where applicable
- fitted-state errors when `predict` is called before `fit`
- invalid input shapes and invalid hyperparameters

Recommended functional tests:
- linear regression on a synthetic line
- logistic regression on a binary dataset
- softmax regression on a multiclass dataset
- Naive Bayes probability checks
- KNN neighbor-based prediction checks
- tree depth and leaf-count checks
- random forest aggregation checks
- boosting staged prediction checks

## Notes
- The focus of this phase is API design and structure, not implementation details.
- The public interface should stay consistent across the library so new algorithms can be added without changing the user experience.
- The naming should stay close to scikit-learn to make the library familiar to users.