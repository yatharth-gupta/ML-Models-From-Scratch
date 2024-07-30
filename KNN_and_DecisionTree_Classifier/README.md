
## KNN and Decision Tree Classifier Implementation

This project implements a K-Nearest Neighbors (KNN) classifier and a Decision Tree classifier using both Powerset and Multi-Output approaches. The project includes data visualization, hyperparameter tuning, performance evaluation, and comparison with scikit-learn's implementations.

### Project Structure

1. **Data Visualization**:
    - Load and visualize the dataset using bar charts, box plots, and pair plots.
    - Example visualizations include gender distribution, income by education, and top most bought items.

2. **KNN Implementation**:
    - Custom KNN class with support for Euclidean, Manhattan, and Cosine distance metrics.
    - Hyperparameter tuning to find the best `k`, encoding, and metric.
    - Performance evaluation using accuracy, precision, recall, and F1-score.
    - Comparison of custom KNN implementation with scikit-learn's KNN in terms of accuracy and inference time.

3. **Decision Tree Classifier**:
    - Implementation of Powerset and Multi-Output Decision Tree classifiers.
    - Preprocessing steps including label encoding and one-hot encoding.
    - Hyperparameter tuning for criterion, max depth, and max features.
    - Performance evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
    - K-Fold cross-validation to evaluate the model's performance.

4. **Optimization**:
    - Discussion on optimizing the execution time using vectorization with NumPy.
    - Comparison of execution time between custom KNN and scikit-learn's KNN.

### How to Run

1. **Data Visualization**:
    - Ensure you have the required dataset (`advertisement.csv`).
    - Run the visualization code to generate plots.

2. **KNN Implementation**:
    - Load the dataset (`data.npy`).
    - Run the KNN implementation and hyperparameter tuning code.
    - Evaluate the performance and compare with scikit-learn's KNN.

3. **Decision Tree Classifier**:
    - Load the dataset (`advertisement.csv`).
    - Preprocess the data and run the Powerset and Multi-Output Decision Tree classifiers.
    - Perform hyperparameter tuning and evaluate the performance.
    - Run K-Fold cross-validation to get average accuracy and precision.

### Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

### Results

- The project provides detailed performance metrics for both KNN and Decision Tree classifiers.
- Visualizations help in understanding the data distribution and feature importance.
- Hyperparameter tuning and cross-validation ensure the models are well-optimized.

### Future Work

- Implement additional distance metrics for KNN.
- Explore other machine learning algorithms for comparison.
- Optimize the code further for large datasets.
