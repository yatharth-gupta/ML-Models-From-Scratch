# Random Forest vs Boosted Trees

## 4.1 - Random Forest

### Classifier

This section implements a custom Random Forest Classifier from scratch using `DecisionTreeClassifier` from `sklearn`. The classifier is tested on the Wine dataset.

#### Implementation Details

- **RandomForest_Classifier**: A class that implements the Random Forest algorithm.
  - **Parameters**:
    - `n_trees`: Number of trees in the forest.
    - `max_depth`: Maximum depth of each tree.
    - `bootstrap`: Whether to use bootstrap samples.
    - `sample_fraction`: Fraction of samples to use for each tree.
  - **Methods**:
    - `fit(X, y)`: Fits the model to the data.
    - `predict(X)`: Predicts the labels for the input data.
    - `bootstrap_sample(X, y)`: Generates bootstrap samples.

#### Results

- **Custom Random Forest Accuracy**: Achieved an accuracy of approximately 0.93 on the Wine dataset.
- **Single Decision Tree Accuracy**: Achieved an accuracy of approximately 0.89.
- **Inbuilt Random Forest Accuracy**: Achieved an accuracy of approximately 0.93.

#### Hyperparameter Tuning

- Best parameters found:
  - `n_trees`: 91
  - `max_depth`: 15
  - `bootstrap`: True
  - `sample_fraction`: 0.75
- Best accuracy: 0.93

### Regression

This section implements a custom Random Forest Regressor from scratch using `DecisionTreeRegressor` from `sklearn`. The regressor is tested on the Boston Housing dataset.

#### Implementation Details

- **RandomForest_Regressor**: A class that implements the Random Forest algorithm for regression.
  - **Parameters**: Same as the classifier.
  - **Methods**: Same as the classifier.

#### Results

- **Custom Random Forest MSE**: Achieved an MSE of approximately 0.23 on the Boston Housing dataset.
- **Single Decision Tree MSE**: Achieved an MSE of approximately 0.26.
- **Inbuilt Random Forest MSE**: Achieved an MSE of approximately 0.23.

#### Hyperparameter Tuning

- Best parameters found:
  - `n_trees`: 500
  - `max_depth`: 25
  - `bootstrap`: True
  - `sample_fraction`: 0.75
- Best MSE: 0.23

## 4.2 - AdaBoost with Decision Trees

### Classifier

This section implements a custom AdaBoost Classifier from scratch using `DecisionTreeClassifier` from `sklearn`. The classifier is tested on the Wine dataset.

#### Implementation Details

- **AdaBoost_Classifier**: A class that implements the AdaBoost algorithm.
  - **Parameters**:
    - `n_estimators`: Number of boosting rounds.
    - `max_depth`: Maximum depth of each tree.
  - **Methods**:
    - `fit(X, y)`: Fits the model to the data.
    - `predict(X)`: Predicts the labels for the input data.

#### Results

- **Best accuracy**: 0.93
- **Best parameters**:
  - `n_estimators`: 91
  - `max_depth`: 1

### Regression

This section implements a custom AdaBoost Regressor from scratch using `DecisionTreeRegressor` from `sklearn`. The regressor is tested on the Boston Housing dataset.

#### Implementation Details

- **AdaBoost_Regressor**: A class that implements the AdaBoost algorithm for regression.
  - **Parameters**: Same as the classifier.
  - **Methods**: Same as the classifier.

#### Results

- **Best MSE**: 0.23
- **Best parameters**:
  - `n_estimators`: 91
  - `max_depth`: 1

## 4.3 - Gradient Boosting

### Classifier

This section implements a custom Gradient Boosting Classifier from scratch using `DecisionTreeRegressor` from `sklearn`. The classifier is tested on the Wine dataset.

#### Implementation Details

- **GradientBoosting_Classifier**: A class that implements the Gradient Boosting algorithm.
  - **Parameters**:
    - `learning_rate`: Learning rate for the boosting process.
    - `max_depth`: Maximum depth of each tree.
    - `n_estimators`: Number of boosting rounds.
  - **Methods**:
    - `fit(X, y)`: Fits the model to the data.
    - `predict(X)`: Predicts the labels for the input data.

#### Results

- **Best accuracy**: 0.93
- **Best parameters**:
  - `n_estimators`: 91
  - `max_depth`: 3
  - `learning_rate`: 0.1

### Regression

This section implements a custom Gradient Boosting Regressor from scratch using `DecisionTreeRegressor` from `sklearn`. The regressor is tested on the Boston Housing dataset.

#### Implementation Details

- **GradientBoosting_Regressor**: A class that implements the Gradient Boosting algorithm for regression.
  - **Parameters**: Same as the classifier.
  - **Methods**: Same as the classifier.

#### Results

- **Best MSE**: 0.23
- **Best parameters**:
  - `n_estimators`: 91
  - `max_depth`: 3
  - `learning_rate`: 0.1

## Analysis of the Mistakes of These Models

These models are sensitive to noisy data and tend to overfit on the noise and outliers. For smaller dimension data, there is a strong reduction of randomness, so random forest would lose its edge over decision trees. Moreover, random forest tends to become slower with the increasing number of trees.

## Feature Similarity

The most striking similarity of all these models is the fact that all of them are prone to noisy labels and outliers. Thus, one needs to prepare quality data before training these models. This is because the classifiers employed by these models are decision trees, which do not handle outliers or noisy data sets.

## Conclusion

This project demonstrates the implementation and comparison of Random Forest, AdaBoost, and Gradient Boosting algorithms for both classification and regression tasks. The results show that these ensemble methods can significantly improve the performance over single decision trees, but they are also sensitive to hyperparameters and data quality.

## Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
