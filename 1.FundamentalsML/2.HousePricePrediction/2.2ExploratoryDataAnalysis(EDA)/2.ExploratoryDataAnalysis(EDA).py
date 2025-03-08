import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ruta del archivo procesado (el que guardamos en DataProcessing.py)
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv'

# Cargar datos
df = pd.read_csv(file_path)

# ==================== HISTOGRAMA ====================
plt.figure(figsize=(10, 6))
plt.hist(df['SalePrice'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Sale Price')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# ==================== BOXPLOT ====================
plt.figure(figsize=(10, 6))
plt.boxplot(df['SalePrice'].dropna(), vert=False)
plt.title('Boxplot of Sale Price')
plt.xlabel('Sale Price')
plt.show()

# ==================== SCATTER PLOT (Gr Liv Area vs Sale Price) ====================
plt.figure(figsize=(10, 6))
plt.scatter(df['Gr Liv Area'], df['SalePrice'], alpha=0.5, color='cornflowerblue', edgecolor='k')
plt.title('Living Area vs Sale Price')
plt.xlabel('Gr Liv Area')
plt.ylabel('Sale Price')
plt.grid(True)
plt.show()

# ==================== MATRIZ DE CORRELACIÓN ====================
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()

plt.xticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, rotation=90, fontsize=8)
plt.yticks(ticks=np.arange(len(correlation_matrix.columns)), labels=correlation_matrix.columns, fontsize=8)
plt.title('Correlation Matrix')
plt.show()

# ==================== LOG TRANSFORMATION (SalePrice) ====================
df['Log_SalePrice'] = np.log1p(df['SalePrice'])

plt.figure(figsize=(10, 6))
plt.hist(df['Log_SalePrice'].dropna(), bins=50, color='lightgreen', edgecolor='black')
plt.title('Histogram of Log Sale Price')
plt.xlabel('Log Sale Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# ==================== LOG SCATTER PLOT ====================
plt.figure(figsize=(10, 6))
plt.scatter(df['Gr Liv Area'], df['Log_SalePrice'], alpha=0.5, color='orange', edgecolor='k')
plt.title('Living Area vs Log Sale Price')
plt.xlabel('Gr Liv Area')
plt.ylabel('Log Sale Price')
plt.grid(True)
plt.show()
