# 0. IMPORT LIBRARIES
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np 
from scipy.stats import shapiro
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# Configuración global de gráficos
plt.rcParams['figure.figsize'] = (10, 6) 

# 1. LOAD DATASET
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/AmesHousing.csv'
df = pd.read_csv(file_path)

# Verificar dimensiones
df_rows, df_columns = df.shape
print(f"Number of rows: {df_rows}")
print(f"Number of columns: {df_columns}")

# Verificar valores nulos y duplicados
null_values = df.isnull().sum()
null_values = null_values[null_values > 0]
print("\nColumns with null values:\n", null_values)

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# 2. EXPLORATORY DATA ANALYSIS (EDA)
df_encoded = pd.get_dummies(df, drop_first=True)

# Correlation Matrix
df_corr = df_encoded.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr, annot=False, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

# Top 10 correlated features with SalePrice
saleprice_corr = df_corr['SalePrice'].abs().sort_values(ascending=False)
print("\nTop 10 Correlated Features with SalePrice:\n", saleprice_corr.head(10))

# Heatmap de características seleccionadas
selected_features = ['SalePrice', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 
                     'Garage Area', 'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Year Built']
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded[selected_features].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of Selected Features')
plt.show()

# Pairplot para visualizar correlaciones
sns.pairplot(df_encoded[selected_features], diag_kind='kde')
plt.show()

# Análisis de valores únicos
print("\nUnique values in selected features:")
for feature in ['Gr Liv Area', 'SalePrice', 'Overall Qual']:
    print(f"{feature}: {df[feature].nunique()} unique values")

# Estadísticas descriptivas
print("\nDescriptive Statistics:")
print(df[['Gr Liv Area', 'SalePrice', 'Overall Qual']].describe())

# Verificación de datos
df['Gr Liv Area'] = pd.to_numeric(df['Gr Liv Area'], errors='coerce')  # Asegurar valores numéricos
df.dropna(subset=['Gr Liv Area'], inplace=True)  # Eliminar nulos

# Histograma de 'Gr Liv Area'
gr_liv_area = df['Gr Liv Area'].to_numpy(dtype=float)
plt.figure(figsize=(8, 6))
sns.histplot(gr_liv_area, kde=True, bins=30)
plt.xlabel("Gr Liv Area")
plt.ylabel("Frequency")
plt.title("Distribution of Gr Liv Area")
plt.show()

# Histograma de 'SalePrice'
sns.histplot(df['SalePrice'], kde=True, bins=30)
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.title("Distribution of SalePrice")
plt.show()

# Histograma de 'Overall Qual'
sns.histplot(df['Overall Qual'], discrete=True)
plt.xlabel("Overall Quality")
plt.ylabel("Count")
plt.title("Distribution of Overall Quality")
plt.show()

plt.figure(figsize = (10,8))
sns.boxplot(y = df ['SalePrice'])
plt.title('Boxplot of SalePrice')
plt.show()

stat, p = shapiro(df['SalePrice'])
print(f'P-value for SalePrice: {p}') 

plt.figure(figsize = (10,8))
sns.boxplot(y = df ['Gr Liv Area'])
plt.title('Boxplot of Gr Liv Area')
plt.show()

stat, p = shapiro(df['Gr Liv Area'])
print(f'P-value for Gr Liv Area: {p}') 

plt.figure(figsize = (10,8))
sns.boxplot(y = df ['Overall Qual'])
plt.title('Boxplot of Overall Qual')
plt.show()

stat, p = shapiro(df['Overall Qual'])
print(f'P-value for Overall Qual: {p}') 

print(df['SalePrice'].skew())
print(df['Gr Liv Area'].skew())
print(df['Overall Qual'].skew())

print()
print(df[df['SalePrice'] <= 0])
print()
print(df[df['Gr Liv Area'] <= 0])
print()
print(df[df['Overall Qual'] <= 0])

print()
print(df[['SalePrice', 'Gr Liv Area', 'Overall Qual']].isnull().sum())

fig,axes = plt.subplots(1, 3, figsize = (18, 5))

sns.histplot(df['SalePrice'], kde = True, bins = 30, ax = axes[0])
axes[0].set_title('Distribution of SalePrice before log')

sns.histplot(df['Gr Liv Area'], kde = True, bins = 30, ax = axes[1])
axes[1].set_title('Distribution of Gr Live Area before log')

sns.histplot(df['Overall Qual'], kde = True, bins = 30, ax = axes[2])
axes[2].set_title('Distribution of Overall Qual before log')

plt.show()

df['SalePrice_log'] = np.log1p(df['SalePrice'])
df['Gr Liv Area_log'] = np.log1p(df['Gr Liv Area'])

fig, axes = plt.subplots(1, 2,  figsize = (12, 5))
sns.histplot(df['SalePrice_log'], kde = True, bins = 30, ax = axes[0])
axes[0].set_title("Distribution of SalePrice after log")

sns.histplot(df['Gr Liv Area_log'], kde = True, bins = 30, ax = axes[1])
axes[1].set_title("Distribution of Gr Liv Area after log")

plt.show()

scaler = MinMaxScaler()
df[['SalePrice_log', 'Gr Liv Area_log']] = scaler.fit_transform(df[['SalePrice_log', 'Gr Liv Area_log']])

print(df[['SalePrice_log', 'Gr Liv Area_log']].head())

fig, axes = plt.subplots(1, 2, figsize = (12, 5))

sns.histplot(df['SalePrice_log'], kde=True, bins = 30, ax = axes[0])
axes[0].set_title("SalePrice after MinMaxScaler")

sns.histplot(df['Gr Liv Area_log'], kde=True, bins = 30, ax = axes[1])
axes[1].set_title("Gr Liv Area after MinMaxScaler")

plt.show()

"""
#-----------------1. KNN REGRESSION----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area_log']]
y = df['SalePrice_log']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

knn1 = KNeighborsRegressor(n_neighbors=5)
knn1.fit(x1_train, y_train)
y_pred_knn1 = knn1.predict(x1_test)

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

knn2 = KNeighborsRegressor(n_neighbors=5)
knn2.fit(x2_train, y_train)
y_pred_knn2 = knn2.predict(x2_test)

#Evaluar Modelos
print()
mse1 = mean_squared_error(y_test, y_pred_knn1)
mse2 = mean_squared_error(y_test, y_pred_knn2)

print(f"MSE for Gr Liv Area: {mse1:.4f}")
print(f"MSE for Overall Qual: {mse2:.4f}") """

"""
#-----------------2. SVR (Support Vector Regressor)----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area_log']]
y = df['SalePrice_log']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

svr1 = SVR(kernel='rbf')
svr1.fit(x1_train, y_train)
y_pred_svr1 = svr1.predict(x1_test)

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

svr2 = SVR(kernel='rbf')
svr2.fit(x2_train, y_train)
y_pred_svr2 = svr2.predict(x2_test)

#Evaluar Modelos
print()
mse1 = mean_squared_error(y_test, y_pred_svr1)
mse2 = mean_squared_error(y_test, y_pred_svr2)

print(f"MSE for Gr Liv Area: {mse1:.4f}")
print(f"MSE for Overall Qual: {mse2:.4f}") 

"""
