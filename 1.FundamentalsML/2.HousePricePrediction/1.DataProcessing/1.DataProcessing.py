import pandas as pd

# Cargar dataset original
file_path = '3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.DataProcessing/AmesHousing.csv'
df = pd.read_csv(file_path)

# Separar numéricas y categóricas
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Rellenar NaN
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna('Missing', inplace=True)

# Guardar limpio (SIN DUMMIES)
cleaned_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.DataProcessing/AmesHousing_cleaned.csv'
df.to_csv(cleaned_path, index=False)

print(f'✅ Dataset limpio guardado en: {cleaned_path}')
