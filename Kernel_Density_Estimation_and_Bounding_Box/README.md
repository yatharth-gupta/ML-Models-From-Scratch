# Kernel Density Estimation (KDE) and Bounding Box Connections

This project implements Kernel Density Estimation (KDE) for both 1D and 2D data using different kernel functions (Gaussian, Box, and Triangular). Additionally, it processes bounding box data from an image, calculates the nearest neighboring boxes, and visualizes the connections between them.

## Table of Contents

1. [KDE Implementation](#kde-implementation)
    - [1D Data Example](#1d-data-example)
    - [2D Data Example](#2d-data-example)
2. [Bounding Box Connections](#bounding-box-connections)
    - [Data Preparation](#data-preparation)
    - [Bounding Box Visualization](#bounding-box-visualization)
    - [Connection Visualization](#connection-visualization)
3. [Gaussian Mixture Model (GMM) for Thresholding](#gaussian-mixture-model-gmm-for-thresholding)
4. [Results](#results)

## KDE Implementation

The KDE implementation includes three types of kernels: Gaussian, Box, and Triangular. The bandwidth selection is performed using a log-likelihood maximization approach.

### 1D Data Example

- Randomly generated 1D data is used to fit the KDE model.
- The density is visualized using Matplotlib.

### 2D Data Example

- Randomly generated 2D data is used to fit the KDE model.
- The density is visualized using a 3D surface plot.

## Bounding Box Connections

### Data Preparation

- Bounding box data is read from a CSV file.
- The coordinates of the top-left and bottom-right corners are extracted and processed to calculate the center points.

### Bounding Box Visualization

- Bounding boxes are drawn on the image using OpenCV.
- Each bounding box is labeled with its ID.

### Connection Visualization

- The nearest neighboring boxes are identified based on Euclidean distance.
- Connections between adjacent boxes (top, bottom, left, right) are visualized using lines.

## Gaussian Mixture Model (GMM) for Thresholding

- GMM is used to set a threshold for the distances between bounding boxes.
- Distances below the mean log-likelihood are considered valid connections, while others are discarded.

## Results

- The KDE implementation successfully visualizes the density of both 1D and 2D data.
- Bounding boxes are accurately drawn and labeled on the image.
- Connections between adjacent bounding boxes are visualized, showing the relationships between them.
- GMM-based thresholding effectively filters out less significant connections, improving the clarity of the visualization.

## Usage

1. **Install Dependencies**:
    - Ensure you have the required libraries installed: `numpy`, `matplotlib`, `scipy`, `pandas`, `opencv-python`, and `scikit-learn`.

2. **Run the KDE Implementation**:
    - Execute the script to fit and visualize KDE for 1D and 2D data.

3. **Process Bounding Box Data**:
    - Load the bounding box data from the CSV file.
    - Visualize the bounding boxes and their connections on the image.

4. **Apply GMM for Thresholding**:
    - Use the GMM-based thresholding function to filter connections.
    - Visualize the final connections on the image.

## Conclusion

This project demonstrates the application of KDE for density estimation and the visualization of bounding box connections using image processing techniques. The use of GMM for thresholding enhances the accuracy of the connections, providing a clear and informative visualization.