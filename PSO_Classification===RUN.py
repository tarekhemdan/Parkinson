import pandas as pd
from pso_numpy import *
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('dataset.csv').drop('name', axis=1)
df.status = np.where(df.status>0.7,1,0)

# Preprocess the dataset
X = df.drop('status', axis=1)
Y = df['status']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the number of nodes in each layer
INPUT_NODES = X_train.shape[1]  # Extract number of features directly from data
HIDDEN_NODES = 100  # You can adjust this based on validation performance
OUTPUT_NODES = 2

def softmax(logits):
    """Calculates the softmax function for normalized probabilities."""
    exps = np.exp(logits)
    return exps / np.sum(exps, axis=1, keepdims=True)

def Negative_Likelihood(probs, Y):
    """Calculates the negative likelihood loss."""
    num_samples = len(probs)
    correct_logprobs = -np.log(probs[range(num_samples), Y])
    return np.sum(correct_logprobs) / num_samples

def forward_pass(X, Y, W):
    """Performs a forward pass through the neural network."""
    if isinstance(W, Particle):
        W = W.x

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
    a1 = np.tanh(z1)  # Tanh activation function
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)
    return Negative_Likelihood(probs, Y)  # Choose appropriate loss function

def predict(X, W):
    """Predicts class labels for new data."""
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
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    logits = z2

    probs = softmax(logits)
    Y_pred = np.argmax(probs, axis=1)
    return Y_pred

def get_accuracy(Y, Y_pred):
    return (Y == Y_pred).mean()

if __name__ == "__main__":
    no_solution = 100
    no_dim = (
        (INPUT_NODES * HIDDEN_NODES)
        + HIDDEN_NODES
        + (HIDDEN_NODES * OUTPUT_NODES)
        + OUTPUT_NODES
    )
    w_range = (0.0, 1.0)
    lr_range = (0.0, 1.0)
    iw_range = (0.9, 0.9)
    c = (0.5, 0.3)

    s = Swarm(no_solution, no_dim, w_range, lr_range, iw_range, c)

    start_time = time.time()

    s.optimize(forward_pass, X_train, Y_train, 100, 1000)  # Use training data for optimization

    end_time = time.time()
    time_consumed = end_time - start_time
    print("Time Consumed: %.3f seconds" % time_consumed)

    W = s.get_best_solution()
    Y_pred_train = predict(X_train, W)
    Y_pred_test = predict(X_test, W)

    # Calculate metrics for both training and testing sets
    train_accuracy = get_accuracy(Y_train, Y_pred_train)
    test_accuracy = get_accuracy(Y_test, Y_pred_test)

    train_precision = precision_score(Y_train, Y_pred_train, average='macro')
    train_recall = recall_score(Y_train, Y_pred_train, average='macro')
    train_f1 = f1_score(Y_train, Y_pred_train, average='macro')
    train_auc = roc_auc_score(Y_train, Y_pred_train)

    test_precision = precision_score(Y_test, Y_pred_test, average='macro')
    test_recall = recall_score(Y_test, Y_pred_test, average='macro')
    test_f1 = f1_score(Y_test, Y_pred_test, average='macro')
    test_auc = roc_auc_score(Y_test, Y_pred_test)

    print("Training Accuracy: %.3f" % train_accuracy)
    print("Training Precision: %.3f" % train_precision)
    print("Training Recall: %.3f" % train_recall)
    print("Training F1 Score: %.3f" % train_f1)
    print("Training AUC Score: %.3f" % train_auc)

    print("Testing Accuracy: %.3f" % test_accuracy)
    print("Testing Precision: %.3f" % test_precision)
    print("Testing Recall: %.3f" % test_recall)
    print("Testing F1 Score: %.3f" % test_f1)
    print("Testing AUC Score: %.3f" % test_auc)

    # Calculate ROC Curve
    fpr_train, tpr_train, thresholds_train = roc_curve(Y_train, Y_pred_train)
    fpr_test, tpr_test, thresholds_test = roc_curve(Y_test, Y_pred_test)

    # Plot ROC Curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label='Train ROC Curve')
    plt.plot(fpr_test, tpr_test, label='Test ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()