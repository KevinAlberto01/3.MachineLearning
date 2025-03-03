import pandas as pd
import numpy as np

# 0. Settings
def print_section(title):
    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50 + "\n")

def dataset_summary(df):
    summary = pd.DataFrame({
        'DataType': df.dtypes,
        'Unique Values': df.nunique(),
        'MissingValues': df.isnull().sum(),
        'MissingPercent': df.isnull().sum() / len(df) * 100
    })
    print("\nDataset Summary:")
    print(summary.sort_values(by='MissingPercent', ascending=False).head(15))  # Show top 15 columns with most missing values

# 1. Load the dataset
print_section("1. Load the dataset")

file_path = "/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.DataProcessing/AmesHousing.csv"
df = pd.read_csv(file_path)

if df.empty:
    print("⚠️ The dataset is empty")
    exit()

# Show first rows
print("First rows of the dataset:")
print(df.head())

# Show general information
print("\nGeneral Info:")
print(df.info())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Missing values check
print("\nMissing Values:")
missing_percent = df.isnull().mean() * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
print(missing_percent)

print("\nMissing values count by column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Initial shape
print(f"\nInitial Shape: {df.shape}")

# 2. Initial Exploring
print_section("2. Initial Exploring")

# Separate columns by type
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical columns ({len(num_cols)}):")
print(num_cols)

print(f"\nCategorical columns ({len(cat_cols)}):")
print(cat_cols)

# 3. Initial Data Cleaning
print_section("3. Cleaning Data")

# Fill numerical values with median
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical values with "Missing"
for col in cat_cols:
    df[col].fillna("Missing", inplace=True)

print("\nMissing values after cleaning (should be 0):")
print(df.isnull().sum().sum())

# 📌 Revisar columnas antes de aplicar dummies
print("\n🔎 Columns before get_dummies:")
print(df.columns)

if 'SalePrice' not in df.columns:
    print("❌ ERROR: SalePrice is missing before encoding! Something is wrong.")
    exit()

# Convert categorical columns to dummies (only low cardinality ones <= 10 unique values)
cat_cols = [col for col in cat_cols if df[col].nunique() <= 10 and col != 'SalePrice']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Validación final: asegurar que SalePrice sigue
if 'SalePrice' not in df.columns:
    print("❌ CRITICAL ERROR: SalePrice disappeared after processing.")
    exit()

# Shape after processing
print(f"\n✅ Shape after cleaning and encoding: {df.shape}")

# 🔎 Check columns before saving
print("\n🔎 Columns before saving:")
print(df.columns)

# Save cleaned dataset
output_path = "/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.ExploratoryDataAnalysis(EDA)/AmesHousing_Cleaned.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Dataset saved at: {output_path}")

# Final dataset summary
dataset_summary(df)

print("\n✅ Data Processing Completed Successfully!")
