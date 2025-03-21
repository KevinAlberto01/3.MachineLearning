import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Load data ---
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

#3. INITIAL DATA SUMMARY
print("Summary of first dates:")
print(df.describe())

# --- Preprocessing: Select only numerical variables ---
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[numeric_cols]

# --- Scale the numeric data ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# --- Apply Jittering ---
def add_jitter(data, noise_level=0.02):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    return data + noise

X_jittered = add_jitter(X_scaled, noise_level=0.02)

# --- Invert the scaling ---
df_jittered = pd.DataFrame(scaler.inverse_transform(X_jittered), columns=numeric_cols)

# --- Handling categorical variables ---
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
df_categorical = df[categorical_cols]

# Take random samples from existing categories for the jittered dataset
df_categorical_jittered = df_categorical.sample(n=df_jittered.shape[0], replace=True, random_state=42).reset_index(drop=True)

# Join numerical and categorical data
df_final_augmented = pd.concat([df_jittered.reset_index(drop=True), df_categorical_jittered], axis=1)

# --- Save augmented dataset ---
output_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Jittering.csv'
df_final_augmented.to_csv(output_path, index=False)

# --- Final message ---
print(f"✅ Jittering completed. File saved at: {output_path}")
print(f"Original size: {df.shape[0]} rows")
print(f"Augmented size: {df_final_augmented.shape[0]} rows")
