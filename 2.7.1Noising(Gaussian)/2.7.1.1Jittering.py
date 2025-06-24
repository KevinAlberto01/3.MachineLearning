#1.IMPORT LIBRARIES
import pandas as pd
import numpy as np

#2.LOADING THE DATASET
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

#3.Display initial summary of data
print("Summary of first dates:")
print(df.describe())

#4.SELECT NUMERIC COLUMNS
numerical_cols = df.select_dtypes(include=[np.number]).columns

#5.DEFINE NOISE LEVEL
# Noise parameter (adjust according to how much noise you want to add)
noise_factor = 0.01 # 1% standard deviation

#6.CREATE AN ENLARGED COPY OF THE DATASET
df_augmented = df.copy()

#7.APPLY GAUSSIAN NOISE TO THE NUMERIC COLUMNS
for col in numerical_cols:
    noise = np.random.normal(loc=0.0, scale=noise_factor * df[col].std(), size=len(df))
    df_augmented[col] += noise

#8.SAVE THE AUGMENTED DATASET
output_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Jittering.csv'
df_augmented.to_csv(output_path, index=False)

print(f"Data augmentation with Gaussian noise completed.  File saved in: {output_path}")

#9.COMPARE THE ORIGINAL AND AUGMENTED DATA
col_to_compare = 'saleprice'
print(f"\nComparing column {col_to_compare} Original vs Augmented (First 5 rows):")
print(pd.DataFrame({
    'Original': df[col_to_compare].head(),
    'Augmented': df_augmented[col_to_compare].head()
}))
