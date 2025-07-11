import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split  # Added missing import
from sklearn.metrics import recall_score, confusion_matrix
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load and prepare data
dataset = pd.read_csv('dataset1.csv').drop('name', axis=1)
X = dataset.drop(['status'], axis=1)
dataset.status = np.where(dataset.status > 0.7, 1, 0)
y = dataset['status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Custom function to calculate specificity
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# Custom classifier that includes sensitivity and specificity
class EnhancedLazyClassifier(LazyClassifier):
    def fit(self, X_train, X_test, y_train, y_test):
        models, predictions = super().fit(X_train, X_test, y_train, y_test)
        
        # Add sensitivity and specificity for each model
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            sensitivity = recall_score(y_test, y_pred)
            specificity = specificity_score(y_test, y_pred)
            
            # Update the models dataframe
            models.loc[name, 'Sensitivity'] = sensitivity
            models.loc[name, 'Specificity'] = specificity
        
        # Reorder columns for better readability
        col_order = ['Accuracy', 'Balanced Accuracy', 'ROC AUC', 'F1 Score', 
                    'Sensitivity', 'Specificity', 'Time Taken']
        models = models[[c for c in col_order if c in models.columns]]
        
        return models, predictions

# Initialize and run enhanced classifier
enhanced_clf = EnhancedLazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = enhanced_clf.fit(X_train, X_test, y_train, y_test)

# Display results
print("Model Performance with Sensitivity and Specificity:")
print(models.sort_values(by='Accuracy', ascending=False))

# Optional: Save results to CSV
models.to_csv('model_performance_with_sensitivity_specificity.csv')
print("\nResults saved to 'model_performance_with_sensitivity_specificity.csv'")