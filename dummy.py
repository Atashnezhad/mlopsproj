import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import shap

# Example: Variable-length original data
data = [
    [2.5, 3.0, 1.5, 2.8],   # Length 4
    [2.7, 3.1, 1.6, 2.9],   # Length 4
    [7.2, 6.8, 5.5, 7.0],   # Length 4
    [7.5, 6.9, 5.7, 7.2]    # Length 4
]
labels = np.array([0, 0, 1, 1])  # Cluster labels

# Convert to NumPy array for processing
data = np.array(data)

# Step 1: Apply PCA for Dimensionality Reduction
pca = PCA(n_components=2)  # Reduce to 2 dimensions
data_pca = pca.fit_transform(data)

# Step 2: Train a Classifier to Detect Feature Importance
clf = RandomForestClassifier()
clf.fit(data_pca, labels)

# Step 3: Use SHAP to Find Discriminating Features in PCA Space
explainer = shap.Explainer(clf, data_pca)
shap_values = explainer(data_pca)

# Step 4: Visualize SHAP Summary in PCA Space
shap.summary_plot(shap_values, data_pca)

# Step 5: Map PCA Importance Back to Original Features
pca_importance = np.abs(pca.components_).T @ clf.feature_importances_

# Normalize importance values
pca_importance /= np.sum(pca_importance)

# Step 6: Identify Which Original Features Contributed to Cluster 1
feature_names = [f"Feature {i+1}" for i in range(data.shape[1])]
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": pca_importance})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# Display feature importance
import ace_tools as tools
tools.display_dataframe_to_user(name="Feature Importance", dataframe=feature_importance_df)