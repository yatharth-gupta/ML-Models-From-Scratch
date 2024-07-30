# Hidden Markov Models for Casino Dice and Cricket Analysis

## Overview

This project involves the application of Hidden Markov Models (HMMs) to two distinct datasets: dice rolls from a casino and cricket runs scored by players. The goal is to decipher hidden states and transitions between them using HMMs.

## Sections

### 1. Casino and Dice

#### 1.1 Dataset

- Loaded a dataset of dice rolls (`rolls.npy`).
- Split the data into training and validation sets, with the first 50% used for training.

#### 1.2 HMM Training

- Generated random emission probabilities for a loaded die.
- Trained multiple HMMs with different emission probabilities.
- Selected the best model based on the highest score on the validation set.

#### 1.3 Results

- **Best Model Parameters**:
  - Emission Probabilities
  - Transition Probabilities
  - Start Probabilities

- **Most Likely Sequence**:
  - Predicted the most likely sequence of hidden states (Fair or Loaded) for the validation data.
  - Visualized the hidden states for the first 200 data points.

#### 1.4 Analysis

- Calculated the number of transitions between fair and loaded dice.
- Analyzed the transition probabilities.
- Visualized the emission probabilities for the loaded die, showing its bias.

### 2. Cricket

#### 2.1 Dataset

- Loaded a dataset of cricket runs (`runs.npy`).
- Replaced all occurrences of 6 with 5 to fit the model requirements.

#### 2.2 HMM Training

- Initialized HMM parameters using the Dirichlet distribution.
- Trained multiple HMMs and selected the best model based on the highest score.

#### 2.3 Results

- **Optimized Model Parameters**:
  - Start Probabilities
  - Transition Probabilities
  - Emission Probabilities

- **Visualizations**:
  - Emission probabilities for Rohit.
  - Transition probabilities heatmap.
  - Start probabilities bar chart.

#### 2.4 Analysis

- Predicted the sequence of players (Virat or Rohit) for each ball.
- Identified the player who played the first and last ball.

## Conclusion

This project demonstrates the application of HMMs to solve the decoding and parameter estimation problems in two different contexts: casino dice rolls and cricket runs. The results provide insights into the hidden states and transitions, helping to understand the underlying patterns in the data.

## Future Work

- Extend the analysis to include more complex models and datasets.
- Explore other machine learning techniques for comparison.
- Improve the visualization and interpretation of results.

## Dependencies

- `numpy`
- `hmmlearn`
- `matplotlib`
- `seaborn`
- `pandas`

## How to Run

1. Install the required dependencies.
2. Load the datasets (`rolls.npy` and `runs.npy`).
3. Execute the scripts to train the models and visualize the results.
