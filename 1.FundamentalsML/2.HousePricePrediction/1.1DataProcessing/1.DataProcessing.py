import pandas as pd

#1.CARGE DATASET
#1.1 Load dataset
file_path = '3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.1DataProcessing/AmesHousing.csv'
df = pd.read_csv(file_path)

#1.2 Separate Numerical and Categorical
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

#1.3 Fill missing values
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna('Missing', inplace=True)

#1.4 Keep it clean (No Dummies)
cleaned_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.1DataProcessing/AmesHousing_cleaned.csv'
df.to_csv(cleaned_path, index=False)

print(f'Clean dataset saved in: {cleaned_path}')
