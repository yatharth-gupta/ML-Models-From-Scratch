# Multi Layer Perceptron Classification

## Overview
This project implements a Multi-Layer Perceptron (MLP) from scratch for classification tasks. The implementation includes both single-label and multi-label classification. The project also demonstrates hyperparameter tuning using Weights & Biases (W&B).

## Dataset
- **Wine Quality Dataset**: Used for single-label classification.
- **Advertisement Dataset**: Used for multi-label classification.

## Dependencies
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `wandb`

## Project Structure
- **Data Preprocessing**: Loading, cleaning, and normalizing the datasets.
- **MLP Implementation**: Custom MLP class for single-label and multi-label classification.
- **Training and Evaluation**: Training the MLP model and evaluating its performance.
- **Hyperparameter Tuning**: Using W&B for hyperparameter tuning.

## Data Preprocessing
### Wine Quality Dataset
1. Load the dataset and drop the `Id` column.
2. Normalize and standardize the data.
3. Partition the dataset into training, validation, and test sets.

### Advertisement Dataset
1. Load the dataset and preprocess categorical and numerical features.
2. Apply label encoding to categorical features.
3. Convert labels to binary vectors using `MultiLabelBinarizer`.
4. Partition the dataset into training and test sets.

## MLP Implementation
### Single-Label Classification
The `MLPClassifier` class is implemented with the following features:
- Forward and backward propagation.
- Activation functions: sigmoid, relu, tanh, and softmax.
- Optimizers: SGD, BGD, and MBGD.
- Loss function: Categorical Cross-Entropy.

### Multi-Label Classification
The `MLPClassifier_multilabel` class is implemented with the following features:
- Forward and backward propagation.
- Activation function: sigmoid.
- Optimizers: SGD, BGD, and MBGD.
- Loss function: Binary Cross-Entropy.

## Training and Evaluation
### Single-Label Classification
1. Initialize the `MLPClassifier` with hyperparameters.
2. Train the model using the training set.
3. Evaluate the model using the test set.
4. Calculate accuracy and generate a classification report.

### Multi-Label Classification
1. Initialize the `MLPClassifier_multilabel` with hyperparameters.
2. Train the model using the training set.
3. Evaluate the model using the test set.
4. Calculate accuracy and generate a classification report.

## Hyperparameter Tuning
1. Initialize W&B and configure the sweep.
2. Define the hyperparameters to tune: learning rate, number of epochs, hidden layers, activation functions, and optimizers.
3. Run the sweep and log the results.

## Results
### Single-Label Classification
- **Accuracy**: Achieved an accuracy of approximately 60% on the test set.
- **Confusion Matrix**: Visualized using a heatmap.
- **Loss Curve**: Plotted the loss curve at intervals of 50 epochs.

### Multi-Label Classification
- **Accuracy**: Achieved an accuracy of approximately 80% on the test set.
- **Predictions**: Thresholded predictions to convert probabilities to binary labels.

## Conclusion
The MLP implementation from scratch demonstrates the ability to perform both single-label and multi-label classification tasks. The project also highlights the importance of hyperparameter tuning to achieve optimal performance.

## Future Work
- Implement additional activation functions and optimizers.
- Experiment with different architectures and datasets.
- Integrate more advanced techniques for hyperparameter tuning.