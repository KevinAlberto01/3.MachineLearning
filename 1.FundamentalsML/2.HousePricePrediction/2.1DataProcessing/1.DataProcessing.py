#1.IMPORT LIBRARIES
import pandas as pd

#2.LOAD DATASET
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.1DataProcessing/AmesHousing.csv'
df = pd.read_csv(file_path)

#3.DATA INSPECTION (Recommended before processing)
print("Dataset Information (Before Cleaning):")
df.info()
print("\nSummary Statistics (Before Cleaning):")
print(df.describe())

#4.COLUMN CLASSIFICATION
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

#5.CLEANING DATA
#5.1 Fill missing values in numerical columns with median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

#5.2 Fill missing values in categorical columns with 'Missing'
for col in cat_cols:
    df[col].fillna('Missing', inplace=True)

#5.3 Column name standardization (optional but recommended)
df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]

#6. SAVE CLEANED DATASET
cleaned_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.1DataProcessing/AmesHousing_cleaned.csv'
df.to_csv(cleaned_path, index=False)
cleaned_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv'
df.to_csv(cleaned_path, index=False)

#7. DATA INSPECTION (After Cleaning)
print("\nDataset Information (After Cleaning):")
df.info()
print("\nSummary Statistics (After Cleaning):")
print(df.describe())
print(f'Clean dataset saved in: {cleaned_path}')
