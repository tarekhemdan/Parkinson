import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # For handling class imbalance
from pso_numpy import *  # Ensure you have the PSO implementation
import time

# Load the dataset
df = pd.read_csv('dataset.csv').drop('name', axis=1)
#df['status'] = np.where(df['status'] > 0.7, 1, 0)  # Binarize the target variable

# Check class distribution
print("Class Distribution:\n", df['status'].value_counts())

# Preprocess the dataset
X = df.drop('status', axis=1)
Y = df['status'].astype(int)  # Ensure the target is integer type for classification

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X, Y = smote.fit_resample(X, Y)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the number of nodes in each layer
INPUT_NODES = X_train.shape[1]  # Number of features
HIDDEN_NODES = 256  # Increased hidden nodes
OUTPUT_NODES = 2  # Binary classification

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Softmax function
def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
    return exps / np.sum(exps, axis=1, keepdims=True)

# Dropout function
def dropout(x, dropout_rate=0.5):
    mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape) / (1 - dropout_rate)
    return x * mask

# Negative log-likelihood with L2 regularization
def Negative_Likelihood(probs, Y, W, lambda_reg=0.01):
    num_samples = len(probs)
    correct_logprobs = -np.log(probs[range(num_samples), Y])
    data_loss = np.sum(correct_logprobs) / num_samples
    reg_loss = 0.5 * lambda_reg * np.sum(W**2)  # L2 regularization
    return data_loss + reg_loss

# Forward pass through the neural network with dropout
def forward_pass(X, Y, W, dropout_rate=0.5):
    if isinstance(W, Particle):
        W = W.x

    # Reshape weights and biases
    w1 = W[0 : INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[
        INPUT_NODES * HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES
    ].reshape((HIDDEN_NODES,))
    w2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
    ].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES) : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    ].reshape((OUTPUT_NODES,))

    # Forward propagation with dropout
    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)  # ReLU activation
    a1 = dropout(a1, dropout_rate)  # Apply dropout
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)
    return Negative_Likelihood(probs, Y, W)  # Include L2 regularization

# Predict class labels
def predict(X, W):
    w1 = W[0 : INPUT_NODES * HIDDEN_NODES].reshape((INPUT_NODES, HIDDEN_NODES))
    b1 = W[
        INPUT_NODES * HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES) + HIDDEN_NODES
    ].reshape((HIDDEN_NODES,))
    w2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
    ].reshape((HIDDEN_NODES, OUTPUT_NODES))
    b2 = W[
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES) : (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    ].reshape((OUTPUT_NODES,))

    z1 = np.dot(X, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)
    Y_pred = np.argmax(probs, axis=1)
    return Y_pred

# Calculate accuracy
def get_accuracy(Y, Y_pred):
    return (Y == Y_pred).mean()

# Calculate sensitivity (recall for positive class) and specificity
def get_sensitivity_specificity(Y_true, Y_pred):
    cm = confusion_matrix(Y_true, Y_pred)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity

# Main execution
if __name__ == "__main__":
    no_solution = 100  # Number of particles in the swarm
    no_dim = (
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    )  # Total number of dimensions
    w_range = (0.0, 1.0)  # Weight range
    lr_range = (0.0, 1.0)  # Learning rate range
    iw_range = (0.9, 0.9)  # Inertia weight range
    c = (0.5, 0.3)  # Cognitive and social coefficients

    # Initialize swarm
    s = Swarm(no_solution, no_dim, w_range, lr_range, iw_range, c)

    # Early stopping parameters
    best_loss = float('inf')
    patience = 10
    no_improvement = 0

    # Training loop with early stopping
    start_time = time.time()
    for epoch in range(1000):  # Maximum epochs
        s.optimize(forward_pass, X_train, Y_train, 100, 1)  # Optimize for 1 epoch
        current_loss = forward_pass(X_test, Y_test, s.get_best_solution())

        if current_loss < best_loss:
            best_loss = current_loss
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    end_time = time.time()
    time_consumed = end_time - start_time
    print("Time Consumed: %.3f seconds" % time_consumed)

    # Get the best solution
    W = s.get_best_solution()

    # Predict on training and testing sets
    Y_pred_train = predict(X_train, W)
    Y_pred_test = predict(X_test, W)

    # Calculate metrics
    train_accuracy = get_accuracy(Y_train, Y_pred_train)
    test_accuracy = get_accuracy(Y_test, Y_pred_test)

    train_precision = precision_score(Y_train, Y_pred_train, average='macro')
    train_recall = recall_score(Y_train, Y_pred_train, average='macro')
    train_f1 = f1_score(Y_train, Y_pred_train, average='macro')
    train_auc = roc_auc_score(Y_train, Y_pred_train)
    train_sensitivity, train_specificity = get_sensitivity_specificity(Y_train, Y_pred_train)

    test_precision = precision_score(Y_test, Y_pred_test, average='macro')
    test_recall = recall_score(Y_test, Y_pred_test, average='macro')
    test_f1 = f1_score(Y_test, Y_pred_test, average='macro')
    test_auc = roc_auc_score(Y_test, Y_pred_test)
    test_sensitivity, test_specificity = get_sensitivity_specificity(Y_test, Y_pred_test)

    # Print metrics
    print("Training Accuracy: %.3f" % train_accuracy)
    print("Training Precision: %.3f" % train_precision)
    print("Training Recall: %.3f" % train_recall)
    print("Training F1 Score: %.3f" % train_f1)
    print("Training AUC Score: %.3f" % train_auc)
    print("Training Sensitivity: %.3f" % train_sensitivity)
    print("Training Specificity: %.3f" % train_specificity)

    print("\nTesting Accuracy: %.3f" % test_accuracy)
    print("Testing Precision: %.3f" % test_precision)
    print("Testing Recall: %.3f" % test_recall)
    print("Testing F1 Score: %.3f" % test_f1)
    print("Testing AUC Score: %.3f" % test_auc)
    print("Testing Sensitivity: %.3f" % test_sensitivity)
    print("Testing Specificity: %.3f" % test_specificity)

    # Confusion matrix and classification report
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(Y_test, Y_pred_test))

    print("\nClassification Report (Test):")
    print(classification_report(Y_test, Y_pred_test))

    # Plot ROC curves
    fpr_train, tpr_train, _ = roc_curve(Y_train, Y_pred_train)
    fpr_test, tpr_test, _ = roc_curve(Y_test, Y_pred_test)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label='Train ROC Curve')
    plt.plot(fpr_test, tpr_test, label='Test ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

    # Save results to file
    with open("PSO_Results.txt", "w") as f:
        f.write("PSO Neural Network Results\n")
        f.write("="*50 + "\n")
        f.write(f"Training Time: {time_consumed:.3f} seconds\n")
        f.write("\nTraining Metrics:\n")
        f.write(f"Accuracy: {train_accuracy:.3f}\n")
        f.write(f"Precision: {train_precision:.3f}\n")
        f.write(f"Recall: {train_recall:.3f}\n")
        f.write(f"F1 Score: {train_f1:.3f}\n")
        f.write(f"AUC Score: {train_auc:.3f}\n")
        f.write(f"Sensitivity: {train_sensitivity:.3f}\n")
        f.write(f"Specificity: {train_specificity:.3f}\n")
        
        f.write("\nTesting Metrics:\n")
        f.write(f"Accuracy: {test_accuracy:.3f}\n")
        f.write(f"Precision: {test_precision:.3f}\n")
        f.write(f"Recall: {test_recall:.3f}\n")
        f.write(f"F1 Score: {test_f1:.3f}\n")
        f.write(f"AUC Score: {test_auc:.3f}\n")
        f.write(f"Sensitivity: {test_sensitivity:.3f}\n")
        f.write(f"Specificity: {test_specificity:.3f}\n")
        
        f.write("\nConfusion Matrix (Test):\n")
        f.write(str(confusion_matrix(Y_test, Y_pred_test)))
        f.write("\n\nClassification Report (Test):\n")
        f.write(classification_report(Y_test, Y_pred_test))