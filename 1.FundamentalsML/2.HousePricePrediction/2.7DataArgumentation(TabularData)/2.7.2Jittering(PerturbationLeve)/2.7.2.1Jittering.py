import pandas as pd
import numpy as np

# --- Load data ---
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

# --- Configure numeric columns for perturbation ---
# We select only numeric columns (we avoid damaging the categorical ones)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Define function to apply jittering (add slight noise)
def add_jitter(df, cols, noise_level=0.01):
    df_jittered = df.copy()
    for col in cols:
        noise = np.random.normal(loc=0.0, scale=noise_level * df[col].std(), size=len(df))
        df_jittered[col] += noise
    return df_jittered

# Apply Jittering
df_jittered = add_jitter(df, numeric_cols, noise_level=0.02)

# Save augmented dataset
output_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.6DataArgumentation(TabularData)/1.6.3Jittering(PerturbationLeve)/AmesHousing_augmented_jittering.csv'
df_jittered.to_csv(output_path, index=False)

# --- Final Messages ---
print(f"✅ Data augmentation with Jittering completed. File saved at: {output_path}")
print(f"Original size: {df.shape[0]} rows")
print(f"Augmented size: {df_jittered.shape[0]} rows (same rows but with noise added)")
