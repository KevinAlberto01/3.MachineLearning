import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# --- Load data ---
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

# --- Select numeric columns ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# We only work with numerical variables (KNN does not handle categorical ones directly)
df_numeric = df[numeric_cols].copy()

# --- Configure the KNN model ---
k = 5  # Number of neighbors
knn = NearestNeighbors(n_neighbors=k)
knn.fit(df_numeric)

# --- Generate new synthetic points ---
synthetic_data = []

for i, point in enumerate(df_numeric.values):
    distances, neighbors = knn.kneighbors([point], n_neighbors=k)
    for j in range(1, k):  # From 1 to avoid the point itself (neighbor 0 is the point itself)
        neighbor = df_numeric.values[neighbors[0][j]]
        synthetic_point = point + np.random.rand() * (neighbor - point)  # Random interpolation
        synthetic_data.append(synthetic_point)

# --- Create augmented DataFrame ---
df_synthetic = pd.DataFrame(synthetic_data, columns=numeric_cols)

# --- Combine with originals ---
df_augmented = pd.concat([df_numeric, df_synthetic], axis=0).reset_index(drop=True)

# --- Add original categorical columns (unmodified) ---
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
df_categorical = df[categorical_cols].copy()

# Repeat the original categorical rows to match with synthetic ones
df_categorical_synthetic = df_categorical.sample(n=len(df_synthetic), replace=True, random_state=42).reset_index(drop=True)

# Unite everything
df_final_augmented = pd.concat([df_augmented, pd.concat([df_categorical, df_categorical_synthetic]).reset_index(drop=True)], axis=1)

# --- Save augmented dataset ---
output_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.6DataArgumentation(TabularData)/1.6.4KNN(SyntheticData)/AmesHousing_augmented_knn.csv'
df_final_augmented.to_csv(output_path, index=False)

# --- Save augmented dataset ---
print(f"✅ Data augmentation with KNN Synthetic Data completed. File saved at: {output_path}")
print(f"Original size: {df.shape[0]} rows")
print(f"Augmented size: {df_final_augmented.shape[0]} rows")
