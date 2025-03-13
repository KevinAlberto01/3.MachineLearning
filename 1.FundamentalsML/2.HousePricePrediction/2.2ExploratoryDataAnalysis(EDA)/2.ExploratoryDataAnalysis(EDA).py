import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo procesado
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv'

# Cargar datos
df = pd.read_csv(file_path)

# Eliminar espacios en los nombres de las columnas
df.columns = df.columns.str.strip()
print(df.columns)

# ==================== RESUMEN ESTADÍSTICO ====================
print("Resumen Estadístico:")
print(df.describe())

# ==================== HISTOGRAMA ====================
plt.figure(figsize=(10, 6))
plt.hist(df['saleprice'].dropna(), bins=50, color='#87CEEB', edgecolor='black')  # Sky Blue
plt.title('Histogram of Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==================== BOXPLOT ====================
plt.figure(figsize=(10, 6))
plt.boxplot(df['saleprice'].dropna(), vert=False, patch_artist=True, boxprops=dict(facecolor='#87CEEB'))
plt.title('Boxplot of Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('Sale Price')
plt.show()

# ==================== SCATTER PLOT (Gr Liv Area vs Sale Price) ====================
plt.figure(figsize=(10, 6))
plt.scatter(df['gr_liv_area'], df['saleprice'], alpha=0.5, color='#4682B4', edgecolor='black')  # Steel Blue
plt.title('Living Area vs Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('gr_liv_area')
plt.ylabel('Sale Price') 
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==================== MATRIZ DE CORRELACIÓN ====================
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')  # Volvemos al estilo original
plt.colorbar()
plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, fontsize=8)
plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.show()

# ==================== LOG TRANSFORMATION (SalePrice) ====================
df['Log_SalePrice'] = np.log1p(df['saleprice'])

plt.figure(figsize=(10, 6))
plt.hist(df['Log_SalePrice'].dropna(), bins=50, color='#00BFFF', edgecolor='black')  # Deep Sky Blue
plt.title('Histogram of Log Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('Log Sale Price')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==================== LOG SCATTER PLOT ====================
plt.figure(figsize=(10, 6))
plt.scatter(df['gr_liv_area'], df['Log_SalePrice'], alpha=0.5, color='#1E90FF', edgecolor='black')  # Dodger Blue
plt.title('Living Area vs Log Sale Price', fontsize=14, fontweight='bold')
plt.xlabel('gr_liv_area')
plt.ylabel('Log Sale Price')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# ==================== CATEGORICAL DATA ANALYSIS ====================
# Asegurarse de que no haya valores nulos en 'House Style'
df = df.dropna(subset=['house_style'])

# Boxplot de precios por estilo de vivienda
plt.figure(figsize=(10, 6))
sns.boxplot(x='house_style', y='saleprice', data=df, palette='Blues')
plt.xticks(rotation=45)
plt.title('Sale Price by House Style', fontsize=14, fontweight='bold')
plt.show()

# Boxplot de precios por vecindario
plt.figure(figsize=(12, 6))
sns.boxplot(x='neighborhood', y='saleprice', data=df, palette='Blues')
plt.xticks(rotation=90)
plt.title('Sale Price by Neighborhood', fontsize=14, fontweight='bold')
plt.show()

# Conteo de categorías clave
plt.figure(figsize=(8, 5))
sns.countplot(x='overall_qual', data=df, hue='overall_qual', palette='Blues', legend=False)
plt.title('Count of Overall Quality', fontsize=14, fontweight='bold')
plt.xlabel('Overall Quality')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
