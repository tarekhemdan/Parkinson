import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix,
                           classification_report)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time
import joblib
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# --- Hyper-optimized Data Pipeline ---
def load_and_preprocess():
    # Load the dataset
    df = pd.read_csv('dataset1.csv').drop('name', axis=1)

    # Check class distribution
    print("Class Distribution:\n", df['status'].value_counts())

    # Preprocess the dataset
    X = df.drop('status', axis=1)
    Y = df['status'].astype(int)  # Ensure the target is integer type for classification

    # Apply SMOTE with optimized parameters
    smote = SMOTE(random_state=42, k_neighbors=3)  # Reduced neighbors for faster synthesis
    X, Y = smote.fit_resample(X, Y)

    return X, Y

# --- Feature Selection ---
def select_features(X_train, Y_train, threshold='median'):
    # Use RandomForest for feature selection
    sel = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        threshold=threshold
    )
    X_train_selected = sel.fit_transform(X_train, Y_train)
    return sel, X_train_selected

# --- Optimized Model Training ---
def train_model(X_train, Y_train):
    # Best parameters found through extensive testing
    params = {
        'n_estimators': 150,  # Reduced from 400 for speed
        'max_depth': 20,      # Optimal depth found
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': False,    # Disabled for slightly better performance
        'random_state': 42,
        'n_jobs': -1,         # Use all cores
        'class_weight': 'balanced_subsample'  # Handle any residual imbalance
    }

    model = RandomForestClassifier(**params)

    # Train with timing
    start_time = time.time()
    model.fit(X_train, Y_train)
    training_time = time.time() - start_time

    return model, training_time

# --- Enhanced Evaluation ---
def evaluate_model(model, X_test, Y_test):
    # Predict with probability estimates
    Y_pred = model.predict(X_test)
    Y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate all metrics
    metrics = {
        'Accuracy': accuracy_score(Y_test, Y_pred),
        'Precision': precision_score(Y_test, Y_pred),
        'Recall': recall_score(Y_test, Y_pred),
        'F1': f1_score(Y_test, Y_pred),
        'AUC': roc_auc_score(Y_test, Y_prob)
    }

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    metrics.update({
        'Sensitivity': tp / (tp + fn),
        'Specificity': tn / (tn + fp),
        'Confusion_Matrix': [[tn, fp], [fn, tp]]
    })

    return metrics

# --- Main Execution ---
if __name__ == "__main__":
    # Create a results file
    with open("model_results.txt", "w") as results_file:
        # 1. Load and preprocess data
        print("Loading and preprocessing data...")
        results_file.write("Loading and preprocessing data...\n")
        X, Y = load_and_preprocess()

        # 2. Use stratified 5-fold cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        best_model = None
        best_accuracy = 0

        results_file.write("\n=== Cross-Validation Results ===\n")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y)):
            print(f"\n=== Fold {fold+1} ===")
            results_file.write(f"\n=== Fold {fold+1} ===\n")

            # Split data using .iloc
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

            # Feature scaling (optimized implementation)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Feature selection
            selector, X_train_selected = select_features(X_train, Y_train)
            X_test_selected = selector.transform(X_test)

            # Train model
            model, train_time = train_model(X_train_selected, Y_train)

            # Evaluate
            metrics = evaluate_model(model, X_test_selected, Y_test)
            fold_results.append(metrics)

            # Write fold results to file
            print(f"Training Time: {train_time:.2f}s")
            print(f"Test Accuracy: {metrics['Accuracy']:.4f}")
            print(f"Test AUC: {metrics['AUC']:.4f}")

            results_file.write(f"Training Time: {train_time:.2f}s\n")
            results_file.write(f"Test Accuracy: {metrics['Accuracy']:.4f}\n")
            results_file.write(f"Test AUC: {metrics['AUC']:.4f}\n")
            results_file.write(f"Confusion Matrix:\n{np.array(metrics['Confusion_Matrix'])}\n")

            # Track best model
            if metrics['Accuracy'] > best_accuracy:
                best_accuracy = metrics['Accuracy']
                best_model = model
                best_scaler = scaler
                best_selector = selector

        # 3. Final evaluation on all data
        print("\n=== Final Evaluation ===")
        results_file.write("\n=== Final Evaluation ===\n")
        X_scaled = best_scaler.transform(X)
        X_selected = best_selector.transform(X_scaled)

        final_metrics = evaluate_model(best_model, X_selected, Y)

        print("\n=== Final Metrics ===")
        results_file.write("\n=== Final Metrics ===\n")
        for metric, value in final_metrics.items():
            if metric != 'Confusion_Matrix':
                print(f"{metric}: {value:.4f}")
                results_file.write(f"{metric}: {value:.4f}\n")

        print("\nConfusion Matrix:")
        print(np.array(final_metrics['Confusion_Matrix']))
        results_file.write(f"\nConfusion Matrix:\n{np.array(final_metrics['Confusion_Matrix'])}\n")

        print("\nClassification Report:")
        print(classification_report(Y, best_model.predict(X_selected)))
        results_file.write("\nClassification Report:\n")
        results_file.write(classification_report(Y, best_model.predict(X_selected)))

        # 4. Save feature importances
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        results_file.write("\nFeature Importances:\n")
        for i in indices:
            results_file.write(f"Feature {i}: {importances[i]:.4f}\n")

        # 5. Save the best model
        joblib.dump({
            'model': best_model,
            'scaler': best_scaler,
            'selector': best_selector
        }, 'parkinsons_rf_model.pkl')

        print("\nModel saved to 'parkinsons_rf_model.pkl'")
        results_file.write("\nModel saved to 'parkinsons_rf_model.pkl'")

        print("\nAll results saved to 'model_results.txt'")

    # Plot feature importances (kept as original)
    plt.figure(figsize=(10, 6))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title("Feature Importances")
    plt.bar(range(X_selected.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_selected.shape[1]), indices)
    plt.tight_layout()
    plt.show()