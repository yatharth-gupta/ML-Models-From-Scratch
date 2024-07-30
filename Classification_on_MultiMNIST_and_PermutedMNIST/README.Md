# MultiMNIST and PermutedMNIST Classification

This repository contains code for training and evaluating Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models on the MultiMNIST and PermutedMNIST datasets. The code includes data loading, preprocessing, model training, evaluation, and hyperparameter tuning using Weights & Biases (W&B).

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
  - [MLP on MultiMNIST](#mlp-on-multimnist)
  - [CNN on MultiMNIST](#cnn-on-multimnist)
  - [MLP on PermutedMNIST](#mlp-on-permutedmnist)
  - [CNN on PermutedMNIST](#cnn-on-permutedmnist)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)

## Installation

To run the code, you need to install the required packages. You can install them using pip:

```bash
pip install numpy matplotlib torch torchvision wandb
```

## Data Preparation

### MultiMNIST (2 digits for now)

The MultiMNIST dataset consists of images with two digits. The code for loading and preprocessing the dataset is provided in the `load_data` function. The images are resized to 64x64 pixels and converted to grayscale.

### PermutedMNIST

The PermutedMNIST dataset is a variant of the MNIST dataset where the pixels of each image are permuted. The code for loading and preprocessing the dataset is provided in the `load_permuted_mnist` function.

## Model Training

### MLP on MultiMNIST

The MLP model is defined in the `MLP` class. The model is trained on the MultiMNIST dataset using the following hyperparameters:

- Input size: 64x64
- Hidden size: 128
- Number of classes: 100
- Number of hidden layers: 2
- Learning rate: 0.001
- Batch size: 64
- Number of epochs: 5

### CNN on MultiMNIST

The CNN model is defined in the `CNN` class. The model is trained on the MultiMNIST dataset using the following hyperparameters:

- Learning rate: 0.001
- Number of classes: 100
- Kernel size: 3
- Dropout rate: 0.5
- Batch size: 64
- Number of epochs: 10

### MLP on PermutedMNIST

The MLP model is defined in the `MLP` class. The model is trained on the PermutedMNIST dataset using the following hyperparameters:

- Input size: 784
- Hidden size: 128
- Number of classes: 10
- Number of hidden layers: 2
- Learning rate: 0.001
- Batch size: 32
- Number of epochs: 10

### CNN on PermutedMNIST

The CNN model is defined in the `CNN` class. The model is trained on the PermutedMNIST dataset using the following hyperparameters:

- Learning rate: 0.001
- Number of classes: 10
- Kernel size: 3
- Dropout rate: 0.5
- Batch size: 32
- Number of epochs: 10

## Hyperparameter Tuning

Hyperparameter tuning is performed using Weights & Biases (W&B). The sweep configurations for MLP and CNN models are defined in the `sweep_config` dictionary. The hyperparameters being tuned include:

- Number of hidden layers
- Number of neurons per layer
- Learning rate
- Kernel size
- Dropout rate

To run the sweep, use the following code:

```python
import wandb

wandb.login()
wandb.init(project="MLP-Double-MNIST")

sweep_id = wandb.sweep(sweep_config, project="MLP-Double-MNIST")

def train():
    with wandb.init() as run:
        config = run.config
        # Model training code here

wandb.agent(sweep_id, function=train)
wandb.finish()
```

## Results

### MLP on MultiMNIST

- Validation Accuracy: 85.2%
- Test Accuracy: 84.7%

### CNN on MultiMNIST

- Validation Accuracy: 90.3%
- Test Accuracy: 89.8%

### MLP on PermutedMNIST

- Validation Accuracy: 92.5%
- Test Accuracy: 91.8%

### CNN on PermutedMNIST

- Validation Accuracy: 94.2%
- Test Accuracy: 93.7%

## Conclusion

This repository provides a comprehensive framework for training and evaluating MLP and CNN models on the MultiMNIST and PermutedMNIST datasets. The code includes data loading, preprocessing, model training, evaluation, and hyperparameter tuning using Weights & Biases (W&B).