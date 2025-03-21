#1.IMPORT LIBRARIES
import pandas as pd
import numpy as np

#2.DATA LOADING
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

#3.INITIAL DATA SUMMARY
print("Summary of first dates:")
print(df.describe())

#4.SELECTION OF NUMERIC COLUMNS
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

#5.DEFINITION OF THE JITTERING FUNCTION
def add_jitter(df, cols, noise_level=0.01):
    df_jittered = df.copy()
    for col in cols:
        noise = np.random.normal(loc=0.0, scale=noise_level * df[col].std(), size=len(df))
        df_jittered[col] += noise
    return df_jittered

#6.APPLICATION OF JITTERING
df_augmented = add_jitter(df, numeric_cols, noise_level=0.02)

#7.COMPARISON BETWEEN ORIGINAL AND AUGMENTED DATA
col_to_compare = 'saleprice'
print(f"\nComparing column {col_to_compare} Original vs Augmented (First 5 rows):")
print(pd.DataFrame({
    'Original': df[col_to_compare].head(),
    'Augmented': df_augmented[col_to_compare].head()
}))

#8.SAVING THE AUGMENTED DATASET
output_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Jittering.csv'
df_augmented.to_csv(output_path, index=False)

#9.FINAL MESSAGES
print(f"✅ Data augmentation with Jittering completed. File saved at: {output_path}")
print(f"Original size: {df.shape[0]} rows")
print(f"Augmented size: {df_augmented.shape[0]} rows (same rows but with noise added)")
