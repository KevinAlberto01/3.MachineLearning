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
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evalute_model(model, x_test, y_test, y_pred, model_name, feature):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {model_name} for {feature}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print("-" * 50)

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

evalute_model(knn1, x1_test, y_test, y_pred_knn1,  "KNN Regressor", "Gr Live Area_log")

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

knn2 = KNeighborsRegressor(n_neighbors=5)
knn2.fit(x2_train, y_train)
y_pred_knn2 = knn2.predict(x2_test)

#Evaluar Modelos
print()
#mse1 = mean_squared_error(y_test, y_pred_knn1)
#mse2 = mean_squared_error(y_test, y_pred_knn2)

#print(f"MSE for Gr Liv Area: {mse1:.4f}")
#print(f"MSE for Overall Qual: {mse2:.4f}") 

evalute_model(knn2, x2_test, y_test, y_pred_knn2,  "KNN Regressor", "Overall Qual")

"""

"""
#-----------------2. SVR (Support Vector Regressor)----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area_log']]
y = df['SalePrice_log']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

svr1 = SVR(kernel='rbf')
svr1.fit(x1_train, y_train)
y_pred_svr1 = svr1.predict(x1_test)

evalute_model(svr1, x1_test, y_test , y_pred_svr1, "SVR", "Gr Liv Area_log")

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

svr2 = SVR(kernel='rbf')
svr2.fit(x2_train, y_train)
y_pred_svr2 = svr2.predict(x2_test)

#Evaluar Modelos
print()
#mse1 = mean_squared_error(y_test, y_pred_svr1)
#mse2 = mean_squared_error(y_test, y_pred_svr2)

#print(f"MSE for Gr Liv Area: {mse1:.4f}")
#print(f"MSE for Overall Qual: {mse2:.4f}") 

evalute_model(svr2, x2_test, y_test , y_pred_svr2, "SVR", "Overall Qual")
#"""

"""
#-----------------3. Redes Neuronales (MLP)----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area_log']]
y = df['SalePrice_log']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

mlp1 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp1.fit(x1_train, y_train)
y_pred_mlp1 = mlp1.predict(x1_test)

evalute_model(mlp1, x1_test, y_test, y_pred_mlp1, "MLP Regressor", "Gr Liv Area_log")

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

mlp2 = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp2.fit(x2_train, y_train)
y_pred_mlp2 = mlp2.predict(x2_test)

#Evaluar Modelos
print()
#mse1 = mean_squared_error(y_test, y_pred_mlp1)
#mse2 = mean_squared_error(y_test, y_pred_mlp2)

#print(f"MSE for Gr Liv Area: {mse1:.4f}")
#print(f"MSE for Overall Qual: {mse2:.4f}") 

evalute_model(mlp2, x2_test, y_test,  y_pred_mlp2, "MLP Regressor", "Overall Qual")

"""

"""
#----------------- 4.Ridge Regression (L2 Regularization) ----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area_log']]
y = df['SalePrice_log']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

rr1 = Ridge(alpha = 1.0)
rr1.fit(x1_train, y_train)
y_pred_rr1 = rr1.predict(x1_test)

evalute_model(rr1, x1_test, y_test, y_pred_rr1, "Ridge Regressor", "Gr Liv Area_log")   

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

rr2 = Ridge(alpha = 1.0)
rr2.fit(x2_train, y_train)
y_pred_rr2 = rr2.predict(x2_test)

#Evaluar Modelos
print()
#mse1 = mean_squared_error(y_test, y_pred_rr1)
#mse2 = mean_squared_error(y_test, y_pred_rr2)

#print(f"MSE for Gr Liv Area: {mse1:.4f}")
#print(f"MSE for Overall Qual: {mse2:.4f}") 

evalute_model(rr2, x2_test, y_test, y_pred_rr2, "Ridge Regressor", "Overall Qual" )

"""

"""
#----------------- 5.Laso Regression (L1 Regularization) ----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area_log']]
y = df['SalePrice_log']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

rr1 = Lasso(alpha = 1.0)
rr1.fit(x1_train, y_train)
y_pred_rr1 = rr1.predict(x1_test)

evalute_model(rr1, x1_test, y_test, y_pred_rr1, "Lasso Regressor", "Gr Liv Area_log")

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

rr2 = Lasso(alpha = 1.0)
rr2.fit(x2_train, y_train)
y_pred_rr2 = rr2.predict(x2_test)

#Evaluar Modelos
print()
#mse1 = mean_squared_error(y_test, y_pred_rr1)
#mse2 = mean_squared_error(y_test, y_pred_rr2)

#print(f"MSE for Gr Liv Area: {mse1:.4f}")
#print(f"MSE for Overall Qual: {mse2:.4f}") 

evalute_model(rr2, x2_test, y_test, y_pred_rr2, "Lasso Regressor", "Overall Qual" )

"""

"""
#----------------- 6.XGBost ----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area_log']]
y = df['SalePrice_log']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

xg1 = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xg1.fit(x1_train, y_train)
y_pred_rr1 = xg1.predict(x1_test)

evalute_model(xg1, x1_test, y_test, y_pred_rr1, "XGBoost Regressor", "Gr Liv Area_log")

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

xg2 = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xg2.fit(x2_train, y_train)
y_pred_rr2 = xg2.predict(x2_test)

#Evaluar Modelos
print()
#mse1 = mean_squared_error(y_test, y_pred_rr1)
#mse2 = mean_squared_error(y_test, y_pred_rr2)

#print(f"MSE for Gr Liv Area: {mse1:.4f}")
#print(f"MSE for Overall Qual: {mse2:.4f}") 

evalute_model(xg2, x2_test, y_test, y_pred_rr2, "XGBoost Regressor", "Overall Qual" )
"""

#"""
#----------------- 7.Ligth GBM ----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area_log']]
y = df['SalePrice_log']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

gbm1 = lgb.LGBMRegressor(objective = 'regression', random_state = 42)
gbm1.fit(x1_train, y_train)
y_pred_rr1 = gbm1.predict(x1_test)

evalute_model(gbm1, x1_test, y_test, y_pred_rr1, "LightGBM Regressor", "Gr Liv Area_log")

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

gbm2 = lgb.LGBMRegressor(objective = 'regression', random_state = 42)
gbm2.fit(x2_train, y_train)
y_pred_rr2 = gbm2.predict(x2_test)

#Evaluar Modelos
print()
#mse1 = mean_squared_error(y_test, y_pred_rr1)
#mse2 = mean_squared_error(y_test, y_pred_rr2)

#print(f"MSE for Gr Liv Area: {mse1:.4f}")
#print(f"MSE for Overall Qual: {mse2:.4f}") 

evalute_model(gbm2, x2_test, y_test, y_pred_rr2, "LightGBM Regressor", "Overall Qual" )

#"""

"""
#-----------------1. Regression Lineal Basic ----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area']]
y = df['SalePrice']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

lr1 = LinearRegression()
lr1.fit(x1_train, y_train)
y_pred_knn1 = lr1.predict(x1_test)

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

lr2 = LinearRegression()
lr2.fit(x2_train, y_train)
y_pred_knn2 = lr2.predict(x2_test)

#Evaluar Modelos
print()
mse1 = mean_squared_error(y_test, y_pred_knn1)
mse2 = mean_squared_error(y_test, y_pred_knn2)

print(f"MSE for Gr Liv Area: {mse1:.4f}")
print(f"MSE for Overall Qual: {mse2:.4f}") 

"""
"""
#-----------------2. Decision Tree Regressor ----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area']]
y = df['SalePrice']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

dtr1 = DecisionTreeRegressor(random_state=42)
dtr1.fit(x1_train, y_train)
y_pred_knn1 = dtr1.predict(x1_test)

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

dtr2 = DecisionTreeRegressor(random_state=42)
dtr2.fit(x2_train, y_train)
y_pred_knn2 = dtr2.predict(x2_test)

#Evaluar Modelos
print()
mse1 = mean_squared_error(y_test, y_pred_knn1)
mse2 = mean_squared_error(y_test, y_pred_knn2)

print(f"MSE for Gr Liv Area: {mse1:.4f}")
print(f"MSE for Overall Qual: {mse2:.4f}") 
"""

"""
#-----------------3.Random Forest Regressor ----------------- #

#SOLO PARA Gr Liv Area_log
x1 = df[['Gr Liv Area']]
y = df['SalePrice']

x1_train, x1_test, y_train, y_test = train_test_split(x1, y, test_size = 0.2, random_state = 42)

dfr1 = RandomForestRegressor(n_estimators=100, random_state=42)
dfr1.fit(x1_train, y_train)
y_pred_knn1 = dfr1.predict(x1_test)

#SOLO PARA Overall Qual
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

dfr2 = RandomForestRegressor(n_estimators=100, random_state=42)
dfr2.fit(x2_train, y_train)
y_pred_knn2 = dfr2.predict(x2_test)

#Evaluar Modelos
print()
mse1 = mean_squared_error(y_test, y_pred_knn1)
mse2 = mean_squared_error(y_test, y_pred_knn2)

print(f"MSE for Gr Liv Area: {mse1:.4f}")
print(f"MSE for Overall Qual: {mse2:.4f}") 

"""

