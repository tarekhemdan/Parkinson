from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2, RFE, SelectFromModel, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('dataset_2.csv').drop('name', axis=1)

# Preprocess the dataset
X = df.drop('status', axis=1)
y = df['status']

# Define the target variable and features
target = 'status'
features = [col for col in df.columns if col != target]

# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Compute pairwise correlations between features
correlations = X.corr()

# Train a random forest classifier to estimate feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Print results
print("Pairwise feature correlations:")
print(correlations)
print("\nFeature importances:")
print(importances)

#############################################################################

# Top Features
# 1. Univariate feature selection with ANOVA F-value
fvalue_selector = SelectKBest(f_classif, k=5)
X_fvalue = fvalue_selector.fit_transform(X, y)

# 2. Univariate feature selection with mutual information
mi_selector = SelectKBest(mutual_info_classif, k=5)
X_mi = mi_selector.fit_transform(X, y)

# 3. Univariate feature selection with chi-square (requires non-negative data)
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)
chi2_selector = SelectKBest(chi2, k=5)
X_chi2 = chi2_selector.fit_transform(X_minmax, y)

# 4. Recursive feature elimination (RFE) with logistic regression
lr = LogisticRegression()
rfe_selector = RFE(lr, n_features_to_select=5)
X_rfe = rfe_selector.fit_transform(X, y)

# 5. Principal component analysis (PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# 6. Feature selection with random forests
rf = RandomForestClassifier()
sfm_selector = SelectFromModel(rf, max_features=5)
X_sfm = sfm_selector.fit_transform(X, y)

# 7. Variance thresholding
vt_selector = VarianceThreshold(threshold=0.1)
X_vt = vt_selector.fit_transform(X)

# 8. Recursive feature elimination with random forests
rfe_rf_selector = RFE(rf, n_features_to_select=5)
X_rfe_rf = rfe_rf_selector.fit_transform(X, y)

# 9. Feature importance with random forests
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
X_fi = X[importances[:5].index]

# Print selected features for each method
print("F-value selector:", X.columns[fvalue_selector.get_support()])
print("=================================================================")

print("Mutual information selector:", X.columns[mi_selector.get_support()])
print("=================================================================")

print("Chi-squared selector:", X.columns[chi2_selector.get_support()])
print("=================================================================")

print("RFE with logistic regression:", X.columns[rfe_selector.get_support()])
print("=================================================================")

print("PCA:", pca.explained_variance_ratio_)
print("=================================================================")

print("Select from model with random forests:", X.columns[sfm_selector.get_support()])
print("=================================================================")

print("Variance thresholding:", X.columns[vt_selector.get_support()])
print("=================================================================")

print("RFE with random forests:", X.columns[rfe_rf_selector.get_support()])
print("=================================================================")

print("Feature importance with random forests:", importances[:5].index)
print("=================================================================")