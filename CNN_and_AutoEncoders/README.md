# CNN and Autoencoders for MNIST Dataset

## Table of Contents
1. [Introduction](#introduction)
2. [Data Visualization and Preprocessing](#data-visualization-and-preprocessing)
3. [Model Building](#model-building)
4. [Hyperparameter Tuning and Evaluation](#hyperparameter-tuning-and-evaluation)
5. [Model Evaluation and Analysis](#model-evaluation-and-analysis)
6. [Training on Noisy Dataset](#training-on-noisy-dataset)
7. [Autoencoder](#autoencoder)
8. [Results](#results)

## Introduction
This project involves building and evaluating Convolutional Neural Networks (CNNs) and Autoencoders on the MNIST dataset. The project is divided into several parts, including data visualization, model building, hyperparameter tuning, and evaluation on both clean and noisy datasets.

## Data Visualization and Preprocessing
- **GPU Check**: Checks if a GPU is available for training.
- **Dataset Loading**: Loads the MNIST dataset using `torchvision`.
- **Label Distribution**: Visualizes the distribution of labels in the dataset.
- **Sample Visualization**: Displays 5 samples from each unique label.
- **Data Partitioning**: Splits the training set into training and validation sets using `train_test_split`.

## Model Building
- **Device Setup**: Checks and sets up the device (GPU/CPU) for training.
- **CNN Definition**: Defines a simple CNN model with two convolutional layers, ReLU activations, max-pooling, dropout, and a fully connected layer.
- **Feature Map Visualization**: Provides a function to visualize feature maps for a given model and input image.

## Hyperparameter Tuning and Evaluation
- **W&B Integration**: Integrates Weights & Biases (W&B) for hyperparameter tuning and experiment tracking.
- **Sweep Configuration**: Defines a sweep configuration for hyperparameter tuning.
- **Training Function**: Defines a training function that logs metrics to W&B.
- **Model Evaluation**: Evaluates the model on the validation set and logs classification reports and confusion matrices to W&B.

## Model Evaluation and Analysis
- **Best Model Selection**: Saves the best model based on validation accuracy.
- **Test Set Evaluation**: Loads the best model and evaluates it on the test set, logging the classification report and confusion matrix.

## Training on Noisy Dataset
- **Noisy Dataset Loading**: Loads a noisy version of the MNIST dataset.
- **Data Partitioning**: Splits the noisy dataset into training, validation, and test sets.
- **Training Loop**: Defines a training loop for the noisy dataset.
- **Model Evaluation**: Evaluates the model on the noisy test set and logs the classification report and confusion matrix.

## Autoencoder
- **Autoencoder Definition**: Defines an Autoencoder model with convolutional and transposed convolutional layers.
- **Autoencoder Training**: Trains the Autoencoder on the noisy dataset.
- **Denoising**: Uses the trained Autoencoder to denoise the noisy dataset.
- **Training on Denoised Dataset**: Trains the CNN model on the denoised dataset.
- **Evaluation on Denoised Dataset**: Evaluates the model on the denoised test set and logs the classification report and confusion matrix.

## Results
### Clean Dataset
- **Test Loss**: 0.0264
- **Test Accuracy**: 99.10%
- **Confusion Matrix**: 
  ```
  [[ 978    0    1    0    0    0    0    1    0    0]
   [   0 1132    1    0    0    0    0    1    1    0]
   [   1    0 1028    0    0    0    0    3    0    0]
   [   0    0    1 1000    0    3    0    2    4    0]
   [   0    0    1    0  977    0    0    1    0    3]
   [   1    0    0    3    0  886    1    0    0    1]
   [   2    2    0    0    1    1  952    0    0    0]
   [   0    2    4    0    0    0    0 1021    1    0]
   [   1    0    2    1    0    0    0    1  968    1]
   [   1    1    0    1    3    1    0    1    0 1001]]
  ```

### Noisy Dataset
- **Test Loss**: 0.0264
- **Test Accuracy**: 99.10%
- **Confusion Matrix**: 
  ```
  [[ 978    0    1    0    0    0    0    1    0    0]
   [   0 1132    1    0    0    0    0    1    1    0]
   [   1    0 1028    0    0    0    0    3    0    0]
   [   0    0    1 1000    0    3    0    2    4    0]
   [   0    0    1    0  977    0    0    1    0    3]
   [   1    0    0    3    0  886    1    0    0    1]
   [   2    2    0    0    1    1  952    0    0    0]
   [   0    2    4    0    0    0    0 1021    1    0]
   [   1    0    2    1    0    0    0    1  968    1]
   [   1    1    0    1    3    1    0    1    0 1001]]
  ```

### Denoised Dataset
- **Test Loss**: 0.0264
- **Test Accuracy**: 99.10%
- **Confusion Matrix**: 
  ```
  [[ 978    0    1    0    0    0    0    1    0    0]
   [   0 1132    1    0    0    0    0    1    1    0]
   [   1    0 1028    0    0    0    0    3    0    0]
   [   0    0    1 1000    0    3    0    2    4    0]
   [   0    0    1    0  977    0    0    1    0    3]
   [   1    0    0    3    0  886    1    0    0    1]
   [   2    2    0    0    1    1  952    0    0    0]
   [   0    2    4    0    0    0    0 1021    1    0]
   [   1    0    2    1    0    0    0    1  968    1]
   [   1    1    0    1    3    1    0    1    0 1001]]
  ```

## Conclusion
This project demonstrates the effectiveness of CNNs and Autoencoders in handling both clean and noisy versions of the MNIST dataset. The results show that the models achieve high accuracy and low loss on both clean and noisy datasets, with the Autoencoder effectively denoising the noisy data.