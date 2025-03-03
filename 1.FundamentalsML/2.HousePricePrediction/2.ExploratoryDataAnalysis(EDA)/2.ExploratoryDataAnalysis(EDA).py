import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETTINGS
def print_section(title):
    print("\n" + "=" * 50)
    print(f"{title}")
    print("=" * 50 + "\n")

# 2. LOAD DATA
print_section("1. Load Cleaned Dataset")
file_path = "/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.ExploratoryDataAnalysis(EDA)/AmesHousing_Cleaned.csv"
df = pd.read_csv(file_path)

# 🔧 Reset index por seguridad (evita MultiIndex o problemas de carga)
df.reset_index(drop=True, inplace=True)

print(f"Dataset loaded successfully! Shape: {df.shape}")
print("🔎 Columns:", df.columns)

# 🔎 Validación rápida de SalePrice
if 'SalePrice' not in df.columns:
    raise ValueError("❌ Column 'SalePrice' not found in the dataset. Check your cleaned CSV file.")

print("🔎 SalePrice index type:", type(df['SalePrice'].index))
print("🔎 Shape de SalePrice:", df['SalePrice'].shape)

# 3. BASIC STATISTICS
print_section("2. Basic Statistics")

print("\nQuick overview of all columns:")
print(df.describe())

print("\nDetailed info about columns:")
print(df.info())

# 4. VISUALIZATION (HISTOGRAM, CORRELATION MATRIX)
print_section("3. Visualization")

# SalePrice histogram
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, color='skyblue')
plt.title('Distribution of Sale Price')
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix')
plt.show()

# 5. Identify Top Correlations with SalePrice
print_section("4. Top Correlations with SalePrice")

correlations = corr_matrix['SalePrice'].sort_values(ascending=False)
print(correlations.head(15))  # Top 15 features

# 6. Optional - Save EDA Report
report_path = "/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.ExploratoryDataAnalysis(EDA)/EDA_Report.txt"

with open(report_path, 'w') as f:
    f.write("Top 15 Correlations with SalePrice:\n")
    f.write(correlations.head(15).to_string())
    f.write("\n\nBasic Statistics:\n")
    f.write(df.describe().to_string())

print(f"\n✅ EDA Report saved at: {report_path}")
