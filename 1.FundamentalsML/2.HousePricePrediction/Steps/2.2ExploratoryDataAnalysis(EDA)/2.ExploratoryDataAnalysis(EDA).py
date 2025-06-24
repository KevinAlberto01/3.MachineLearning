#1.IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#2.LOAD DATASET
#2.1 Path of the processed file
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv'
#2.1 Load the dataset
df = pd.read_csv(file_path)

#3.CLEANING COLUMN NAMES
df.columns = df.columns.str.strip()
print(df.columns)

# ==================== 4.STATISTICAL SUMMARY ====================
print("Resumen Estadístico:")
print(df.describe())

# ==================== 5.HISTOGRAM OF SALE PRICE ====================
plt.figure(figsize=(10, 6))
plt.hist(df['saleprice'].dropna(), bins=50, color='#87CEEB', edgecolor='black')  # Sky Blue
plt.title('Histogram of Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==================== 6.BOXPLOT OF THE SACALE PRICE ====================
plt.figure(figsize=(10, 6))
plt.boxplot(df['saleprice'].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='#87CEEB'))
plt.title('Boxplot of Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('Sale Price')
plt.show()

# ==================== 7.SCATTER PLOT (HOUSING AREA VS SALE PRICE) ====================
plt.figure(figsize=(10, 6))
plt.scatter(df['gr_liv_area'], df['saleprice'], alpha=0.5, color='#4682B4', edgecolor='black')  # Steel Blue
plt.title('Living Area vs Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('gr_liv_area')
plt.ylabel('Sale Price') 
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==================== 8.CORRELATION MATRIX ====================
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')  # Volvemos al estilo original
plt.colorbar()
plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, fontsize=8)
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.show()

# ==================== 9.LOGARITHMIC TRANSFORMATION OF THE SELLING PRICE ====================
df['Log_SalePrice'] = np.log1p(df['saleprice'])

plt.figure(figsize=(10, 6))
plt.hist(df['Log_SalePrice'].dropna(), bins=50, color='#00BFFF', edgecolor='black')  # Deep Sky Blue
plt.title('Histogram of Log Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('Log Sale Price')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==================== 10.SCATTER PLOT WITH LOGARITHMIC PRICE ====================
plt.figure(figsize=(10, 6))
plt.scatter(df['gr_liv_area'], df['Log_SalePrice'], alpha=0.5, color='#1E90FF', edgecolor='black')  # Dodger Blue
plt.title('Living Area vs Log Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('gr_liv_area')
plt.ylabel('Log Sale Price')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==================== 11.ANALYSIS OF CATEGORICAL VARIABLES ====================
#Ensure there are no null values in 'House Style'
df = df.dropna(subset=['house_style'])

#Boxplot of prices by housing style
plt.figure(figsize=(10, 6))
sns.boxplot(x='house_style', y='saleprice', data=df, palette='Blues')
plt.xticks(rotation=45)
plt.title('Sale Price by House Style', fontsize=14, fontweight='bold')
plt.show()

#Boxplot of prices by neighborhood
plt.figure(figsize=(12, 6))
sns.boxplot(x='neighborhood', y='saleprice', data=df, palette='Blues')
plt.xticks(rotation=90)
plt.title('Sale Price by Neighborhood', fontsize=14, fontweight='bold')
plt.show()

#Key category count
plt.figure(figsize=(8, 5))
sns.countplot(x='overall_qual', data=df, hue='overall_qual', palette='Blues', legend=False)
plt.title('Count of Overall Quality', fontsize=14, fontweight='bold')
plt.xlabel('Overall Quality')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
