# PSO-Based Neural Network Training for Parkinson Classification

This repository implements a neural network for binary classification using Particle Swarm Optimization (PSO) for training. The project leverages deep learning techniques along with various preprocessing and evaluation methods to handle imbalanced datasets and achieve robust classification performance.

## Overview

The code performs the following tasks:

- **Data Loading and Preprocessing:**  
  - Loads a dataset from `dataset.csv` (dropping the `name` column).
  - Checks class distribution.
  - Balances classes using SMOTE.
  - Splits the data into training and testing sets.
  - Normalizes features using StandardScaler.

- **Model Architecture:**  
  - A simple neural network with one hidden layer of 256 nodes.
  - Uses ReLU activation in the hidden layer.
  - Applies dropout for regularization.
  - Implements a softmax output layer.
  - Uses negative log-likelihood loss with L2 regularization.

- **Training with PSO:**  
  - Employs the `pso_numpy` package to optimize network weights.
  - Includes early stopping based on test loss.
  - Measures training time.

- **Evaluation:**  
  - Predicts class labels on training and testing sets.
  - Computes accuracy, precision, recall, F1-score, and ROC AUC.
  - Displays a confusion matrix and a classification report.
  - Plots ROC curves for both training and testing sets.

## Dependencies

Make sure you have the following Python packages installed:

- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [matplotlib](https://matplotlib.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [pso_numpy](https://github.com/your_pso_numpy_repo) (or your specific PSO implementation)

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib imbalanced-learn pso_numpy
Usage
Dataset Preparation:
Place your dataset.csv file in the repository root. The file should include:

A name column (which will be dropped).
A status column representing the target variable for binary classification.
Run the Script:
Execute the Python script by running:

bash
Copy
Edit
python your_script_name.py
Replace your_script_name.py with the name of the Python file containing the code.

Output:

The script prints the class distribution, training/testing metrics, confusion matrix, and classification report.
It also plots the ROC curves for both the training and testing datasets.
Code Breakdown
Imports and Dataset Loading:
The script starts by importing necessary libraries and loading the dataset using Pandas.

Data Preprocessing:
SMOTE is applied to address class imbalance, followed by a train-test split and feature scaling.

Neural Network Functions:

relu, softmax, and dropout functions are defined.
A custom loss function (Negative_Likelihood) with L2 regularization is implemented.
The forward_pass function computes the loss using the current weights.
A predict function is provided for inference.
PSO Training Loop:
The PSO algorithm is used to optimize the network's weights. The loop includes:

Optimization for a fixed number of epochs.
Early stopping if the loss does not improve for a specified number of epochs.
Calculation of training time.
Model Evaluation:
After training, the model's performance is evaluated on both training and testing data using various metrics, and ROC curves are plotted.

