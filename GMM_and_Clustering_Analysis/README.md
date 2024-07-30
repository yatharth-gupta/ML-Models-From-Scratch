# Gaussian Mixture Models (GMM) and Clustering Analysis

## Overview

This project involves implementing the Expectation-Maximization (EM) algorithm for Gaussian Mixture Models (GMM) and performing clustering operations on given datasets. The tasks include finding the parameters of GMM for a customer dataset, performing clustering on the wine dataset using both GMM and K-Means algorithms, and determining the optimal number of clusters using Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC). The project also involves reducing the dataset dimensions using Principal Component Analysis (PCA) and comparing the clustering results using silhouette scores.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Plotly
- scikit-learn (for dataset loading, PCA, and silhouette score computation)

## GMM Implementation

The `GMM` class implements the EM algorithm for fitting a Gaussian Mixture Model to the data. Key methods include:

- `fit(data, k, max_iter=20)`: Fits the GMM to the data with `k` clusters and a maximum of `max_iter` iterations.
- `e_step()`: Performs the expectation step of the EM algorithm.
- `m_step()`: Performs the maximization step of the EM algorithm.
- `gaussian(x, mu, sigma)`: Computes the Gaussian probability density function.
- `likelihood()`: Computes the log-likelihood of the data given the current parameters.
- `get_parameters()`: Returns the parameters of the GMM.
- `get_membership_values()`: Returns the membership values of the data samples.

## Clustering Analysis

### Data Loading and Preprocessing

- The customer dataset is loaded and normalized.
- The Wine dataset is loaded and reduced to 2 principal components using PCA for visualization.

### Optimal Number of Clusters

- The optimal number of clusters is determined using BIC and AIC.
- The elbow point is identified where the sum of AIC and BIC stops decreasing significantly.

### Clustering and Visualization

- Both K-Means and GMM are applied to the Wine dataset.
- Scatter plots are generated to visualize the clustering results.

### Performance Evaluation

- Silhouette scores and accuracy scores are computed for both K-Means and GMM to evaluate their performance.

## Results

### Optimal Number of Clusters

- The optimal number of clusters for the Wine dataset is determined to be 3.

### Clustering Performance

- GMM outperforms K-Means in this case because the data is generated from 3 Gaussian distributions and is not linearly separable.

### Silhouette Scores

- Silhouette score for K-Means: `0.57`
- Silhouette score for GMM: `0.65`

### Accuracy Scores

- Accuracy score for K-Means: `0.73`
- Accuracy score for GMM: `0.85`

## Observations and Analysis

1. The optimal number of clusters is 3.
2. GMM is better than K-Means for this dataset.
3. The data is generated from 3 Gaussian distributions.
4. The data is not linearly separable, making K-Means less effective.

## Conclusion

The GMM with the EM algorithm provides a robust method for clustering data, especially when the data is generated from Gaussian distributions. This project demonstrates the effectiveness of GMM over K-Means for the Wine dataset and provides a framework for determining the optimal number of clusters using BIC and AIC.