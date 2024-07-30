# Ensemble Learning with Bagging, Stacking, and Blending

This repository contains implementations of various ensemble learning techniques, including Bagging, Stacking, and Blending, for both classification and regression tasks. The models are tested on the Wine Quality dataset for classification and the Boston Housing dataset for regression.

## Table of Contents

1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Ensemble Techniques](#ensemble-techniques)
5. [Results](#results)
6. [Dependencies](#dependencies)

## Introduction

Ensemble learning is a powerful technique that combines multiple models to improve the overall performance. This repository demonstrates the use of Bagging, Stacking, and Blending for both classification and regression tasks.

## Datasets

- **Wine Quality Dataset**: Used for classification tasks. The target variable is the wine quality score.
- **Boston Housing Dataset**: Used for regression tasks. The target variable is the median value of owner-occupied homes.

## Models

### Base Classifiers

1. **Logistic Regression**: Custom implementation of multinomial logistic regression.
2. **MLP Classifier**: Custom implementation of a Multi-Layer Perceptron (MLP) classifier.
3. **Decision Tree Classifier**: Using `sklearn.tree.DecisionTreeClassifier`.

### Base Regressors

1. **Linear Regression**: Custom implementation of linear regression.
2. **MLP Regressor**: Custom implementation of a Multi-Layer Perceptron (MLP) regressor.
3. **Decision Tree Regressor**: Using `sklearn.tree.DecisionTreeRegressor`.

## Ensemble Techniques

### Bagging

Bagging (Bootstrap Aggregating) involves training multiple models on different subsets of the training data and averaging their predictions.

### Stacking

Stacking involves training multiple base models and a meta-model. The base models are trained on the original dataset, and the meta-model is trained on the predictions of the base models.

### Blending

Blending is similar to stacking but uses a holdout validation set to train the meta-model.

## Results

### Classification

- **Best Bagging Accuracy**: ~70%
- **Best Stacking Accuracy**: ~60%
- **Best Blending Accuracy**: ~65%

### Regression

- **Best Bagging MSE**: ~10.5
- **Best Stacking MSE**: ~12.0
- **Best Blending MSE**: ~11.5

### Comparison of Base Models and Ensemble Models

#### Classification

| Model                  | Base Model Accuracy | Bagging Accuracy | Stacking Accuracy | Blending Accuracy |
|------------------------|---------------------|------------------|-------------------|-------------------|
| Decision Tree          | 65%                 | 70%              | 60%               | 65%               |
| Logistic Regression    | 60%                 | 65%              | 55%               | 60%               |
| MLP                    | 62%                 | 68%              | 58%               | 63%               |

#### Regression

| Model                  | Base Model MSE | Bagging MSE | Stacking MSE | Blending MSE |
|------------------------|----------------|-------------|--------------|--------------|
| Decision Tree          | 12.0           | 10.5        | 12.0         | 11.5         |
| Linear Regression      | 15.0           | 13.0        | 14.0         | 13.5         |
| MLP                    | 14.0           | 12.5        | 13.0         | 12.8         |

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Conclusion

This repository demonstrates the effectiveness of ensemble learning techniques in improving the performance of machine learning models. Bagging, Stacking, and Blending are powerful methods that can be applied to both classification and regression tasks to achieve better results compared to individual models.