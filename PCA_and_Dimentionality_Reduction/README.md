# PCA and Dimentionality Reduction on Image and Pictionary Datasets

## Overview

This project demonstrates the application of Principal Component Analysis (PCA) for dimensionality reduction on two datasets: a celebrity image dataset and a pictionary dataset. The project includes data preprocessing, PCA implementation, and classification accuracy comparison between original and reduced datasets using KNN.

## Requirements

- Python 3.x
- Libraries: `numpy`, `os`, `random`, `matplotlib`, `Pillow`, `scikit-learn`, `pandas`

Install the required libraries using:
```bash
!pip install scikit-learn matplotlib Pillow pandas
```

## Code Summary

### 1. Data Loading and Preprocessing

#### Celebrity Image Dataset

- **Dictionary of Classes**: `cfw_dict` maps celebrity names to numerical labels.
- **Options**: `opt` dictionary contains image processing options like image size, grayscale conversion, and validation split ratio.
- **Functions**:
  - `load_image(path)`: Loads and preprocesses an image.
  - `display_images(imgs, classes, row, col, w, h)`: Displays a grid of images with their respective classes.
  - `load_data(dir_path)`: Loads images from the specified directory, preprocesses them, and returns image and label arrays.

#### Pictionary Dataset

- **Loading**: The dataset is loaded from a `.npy` file.
- **Preprocessing**: Mean normalization is applied to the data.

### 2. PCA Implementation

- **Flattening Data**: The image data is flattened for PCA.
- **Mean Normalization**: The data is mean normalized.
- **Covariance Matrix**: Computed using `np.cov`.
- **Eigen Decomposition**: Eigenvalues and eigenvectors are computed using `np.linalg.eigh`.
- **Sorting**: Eigenvalues and eigenvectors are sorted in descending order of eigenvalues.
- **Cumulative Explained Variance**: Plotted to determine the number of components needed to cover a significant variance.

### 3. KNN Classification

- **Original Data**: KNN classifier is trained and tested on the original dataset.
- **Reduced Data**: KNN classifier is trained and tested on the PCA-reduced dataset for various numbers of components.
- **Accuracy Comparison**: Accuracy of the classifier on original and reduced datasets is plotted.

### 4. Dimensionality Reduction on Pictionary Dataset

- **PCA on Drawer and Guesser Data**: PCA is applied separately to drawer and guesser attributes.
- **Visualization**: PCA components are visualized in 1D, 2D, and 3D plots.

## Results

### Celebrity Image Dataset

- **Cumulative Explained Variance**: The plot shows the variance covered by the first `k` components.
- **KNN Classification**: Accuracy of the classifier on original and reduced datasets is compared. The accuracy becomes almost the same as the original dataset when the number of components is around 350.

### Pictionary Dataset

- **PCA Components Visualization**: The new axes obtained represent the principal components of the data. These components are linear combinations of the original features and are ordered by the amount of variance they explain in the data.

## Observations

- **Dimensionality Reduction**: PCA effectively reduces the dimensionality of the data while retaining most of the variance.
- **Classification Accuracy**: The accuracy of the KNN classifier on the reduced dataset approaches that of the original dataset as the number of components increases.
- **PCA Components**: The principal components provide new axes that are linear combinations of the original features, ordered by the amount of variance they explain.

## Conclusion

This project demonstrates the effectiveness of PCA for dimensionality reduction and its impact on the performance of a KNN classifier. The results show that a significant reduction in dimensionality can be achieved without a substantial loss in classification accuracy.