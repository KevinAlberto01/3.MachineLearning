import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

# Show initial summary
print("Summary of first dates:")
print(df.describe())

# Apply Gaussian noise only to numeric columns (float or int)
numerical_cols = df.select_dtypes(include=[np.number]).columns

# Noise parameter (adjust according to how much noise you want to add)
noise_factor = 0.01 # 1% standard deviation

# Create augmented copy
df_augmented = df.copy()

for col in numerical_cols:
    noise = np.random.normal(loc=0.0, scale=noise_factor * df[col].std(), size=len(df))
    df_augmented[col] += noise

# Save the new augmented dataset
output_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Jittering.csv'
df_augmented.to_csv(output_path, index=False)

print(f"Data augmentation with Gaussian noise completed.  File saved in: {output_path}")

# Compare a column as an example
col_to_compare = 'saleprice'
print(f"\nComparing column {col_to_compare} Original vs Augmented (First 5 rows):")
print(pd.DataFrame({
    'Original': df[col_to_compare].head(),
    'Augmented': df_augmented[col_to_compare].head()
}))
