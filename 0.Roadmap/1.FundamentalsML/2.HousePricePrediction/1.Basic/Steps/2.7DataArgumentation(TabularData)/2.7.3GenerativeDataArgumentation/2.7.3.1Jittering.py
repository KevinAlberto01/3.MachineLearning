#1.IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#2.LOAD DATA FROM CSV FILE
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

#3.SHOW A STATISTICAL SUMMARY OF THE DATA
print("Summary of first dates:")
print(df.describe())

#4.SELECT ONLY THE NUMERICAL VARIABLES
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_numeric = df[numeric_cols]

#5.SCALE THE DATA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

#6.APPLY JITTERING (ADD NOISE)
def add_jitter(data, noise_level=0.02):
    noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape)
    return data + noise

X_jittered = add_jitter(X_scaled, noise_level=0.02)

#7.RESTORE THE VALUES TO THEIR ORIGINAL SCALE
df_jittered = pd.DataFrame(scaler.inverse_transform(X_jittered), columns=numeric_cols)

#8.HANDLING CATEGORICAL VARIABLES
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
df_categorical = df[categorical_cols]

#8.1 Take random samples from existing categories for the jittered dataset
df_categorical_jittered = df_categorical.sample(n=df_jittered.shape[0], replace=True, random_state=42).reset_index(drop=True)

#9.COMBINE NUMERICAL AND CATEGORICAL VARIABLES
df_final_augmented = pd.concat([df_jittered.reset_index(drop=True), df_categorical_jittered], axis=1)

#10.SAVE THE AUGMENTED DATASET
output_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Jittering.csv'
df_final_augmented.to_csv(output_path, index=False)

#11.FINAL MESSAGES
print(f"✅ Jittering completed. File saved at: {output_path}")
print(f"Original size: {df.shape[0]} rows")
print(f"Augmented size: {df_final_augmented.shape[0]} rows")
