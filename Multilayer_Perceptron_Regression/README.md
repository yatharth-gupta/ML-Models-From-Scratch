# Multilayer Perceptron Regression

## Overview
The project involves creating an MLP from scratch using Python and NumPy to perform regression on a housing dataset. The process includes data preprocessing, model training, and hyperparameter tuning using Weights & Biases (W&B).

### Data Preprocessing
1. **Import Libraries**: Essential libraries like NumPy, Pandas, Matplotlib, and Scikit-learn are imported.
2. **Load and Preprocess Data**:
   - Load the dataset.
   - Handle missing values by replacing them with the mean of the respective columns.
   - Calculate and display statistical metrics (mean, standard deviation, minimum, and maximum) for each feature.
   - Plot the feature values.
   - Split the data into training, validation, and test sets.
   - Standardize the data using `StandardScaler`.

### MLP Regression Implementation
1. **Define MLPRegressor Class**:
   - Initialize weights and biases.
   - Implement activation functions (sigmoid, ReLU, tanh) and their derivatives.
   - Implement forward and backward propagation.
   - Implement training methods: Batch Gradient Descent (BGD), Stochastic Gradient Descent (SGD), and Mini-Batch Gradient Descent (MBGD).
   - Implement prediction method.

2. **Train and Evaluate the Model**:
   - Train the model using the training set.
   - Evaluate the model using the validation set.
   - Calculate and print the validation loss.

### Hyperparameter Tuning
1. **Setup Weights & Biases**:
   - Install and login to W&B.
   - Define the sweep configuration for hyperparameter tuning.
   - Initialize the sweep.

2. **Train Function for Sweep**:
   - Define the training function to be used by the sweep agent.
   - Log the mean squared error (MSE) for each run.

### Final Model Evaluation
1. **Train and Evaluate the Final Model**:
   - Train the model using the best hyperparameters.
   - Evaluate the model using the test set.
   - Calculate and print the test loss, RMSE, and R-squared value.

### Results
- **Validation Loss**: Calculated during the validation phase.
- **Test Loss**: Calculated during the test phase.
- **RMSE**: Root Mean Squared Error for the test set.
- **R-Squared**: R-squared value for the test set.

### Conclusion
The project demonstrates the implementation of an MLP for regression tasks from scratch, including data preprocessing, model training, and hyperparameter tuning using W&B. The final model is evaluated on a test set to determine its performance.

### Dependencies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Weights & Biases

### How to Run
1. Clone the repository.
2. Install the required dependencies.
3. Run the script to preprocess the data, train the model, and evaluate its performance.
