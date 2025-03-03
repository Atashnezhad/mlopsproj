import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import shap

# Example: Variable length data (each row is a different sample)
data = [
    [2.5, 3.0, 1.5, 2.8],   # Length 4
    [2.7, 3.1, 1.6],        # Length 3
    [7.2, 6.8, 5.5, 7.0],   # Length 4
    [7.5, 6.9]              # Length 2
]

labels = np.array([0, 0, 1, 1])  # Cluster labels

# Step 1: Normalize to Minimum Length
min_length = min(len(row) for row in data)  # Find shortest length
data_fixed = np.array([row[:min_length] for row in data])  # Truncate to min length

# Step 2: Apply PCA for Dimensionality Reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
data_pca = pca.fit_transform(data_fixed)

# Step 3: Train a Classifier to Detect Feature Importance
clf = RandomForestClassifier()
clf.fit(data_pca, labels)

# Step 4: Use SHAP to Find Discriminating Features
explainer = shap.Explainer(clf, data_pca)
shap_values = explainer(data_pca)

# Visualize SHAP Summary Plot
shap.summary_plot(shap_values, data_pca)