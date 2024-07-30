# Multinomial Logistic Regression

## 1.1 - Dataset Analysis and Preprocessing

### Dataset Loading and Initial Analysis
- The dataset `WineQT.csv` is loaded using pandas.
- The `Id` column is dropped as it is not needed for the analysis.
- Basic statistics (mean, standard deviation, minimum, and maximum values) for each feature are calculated and displayed.
- The frequency distribution of the `quality` column is plotted.

### Data Partitioning
- The dataset is partitioned into training, validation, and test sets using an 80-20 split for training and test sets, and a further 80-20 split for training and validation sets.

### Data Normalization
- The features are normalized using `StandardScaler` from `sklearn`.

## 1.2 - Model Building from Scratch

### Multinomial Logistic Regression Class
- A custom class `MultiLogisticRegression` is implemented to perform multinomial logistic regression.
- The class includes methods for fitting the model (`fit`), predicting (`predict`), calculating softmax probabilities (`softmax`), computing log loss (`log_loss`), and calculating accuracy (`acc`).

### Model Training and Evaluation
- The model is trained on the training set and evaluated on the validation set.
- Loss and accuracy are logged at every 100 epochs.
- The final classification report for the validation set is printed.
- Loss and accuracy curves for the training and validation sets are plotted.

### Hyperparameter Tuning
- Hyperparameter tuning is performed using Weights & Biases (wandb) with a grid search over learning rates and number of epochs.
- The best hyperparameters are selected based on validation accuracy.

### Final Model Evaluation
- The model is retrained using the best hyperparameters and evaluated on the test set.
- The final classification report for the test set is printed.

## Results
- The final classification report for the test set is provided, showing the performance of the model on unseen data.

## Conclusion
- The custom multinomial logistic regression model was successfully implemented and evaluated.
- Hyperparameter tuning was performed using Weights & Biases to find the best learning rate and number of epochs.
- The final model's performance was evaluated on the test set, and the results were satisfactory.

## How to Run
1. Clone the repository.
2. Install the required libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, and `wandb`.
3. Load the dataset `WineQT.csv` in the same directory as the script.
4. Run the script to train and evaluate the model.
