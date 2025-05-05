#0.IMPORT LIBRARIES 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna 
from lightgbm import early_stopping, log_evaluation
import joblib

#-------------1. LOAD DATASET-------------#
#1.1 Carga de Datos
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/AmesHousing.csv'
df = pd.read_csv(file_path)

df_rows, df_columns = df.shape
#"""
print(f"Number of rows: {df_rows}")
print(f"Number of columns: {df_columns}")

print(df.head())

#1.2 Verificar valores nulos y duplicados
null_values = df.isnull().sum()
null_values = null_values[null_values > 0]
print("\nColumns with null values:\n", null_values)

#1.3 Identificacion de datos duplicados
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

#1.4 Visualizacion de tipo de datos 
print(df.info())#"""
#---------------------------------------#

#---2.EXPLORATORY DATA ANALYSIS (EDA)---#
df_encoded = pd.get_dummies(df, drop_first=True)

"""
#2.1.1 Correlation Matrix
df_corr = df_encoded.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(df_corr, annot=False, cmap="coolwarm")
plt.title('Correlation Matrix')
plt.show()

#2.1.2 Top 10 correlated features with SalePrice
saleprice_corr = df_corr['SalePrice'].abs().sort_values(ascending=False)
print("\nTop 10 Correlated Features with SalePrice:\n", saleprice_corr.head(10))

#2.2 Heatmap de características seleccionadas
selected_features = ['SalePrice', 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 
                     'Garage Area', 'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Year Built']
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded[selected_features].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix of Selected Features')
plt.show()

#2.3 Pairplot para visualizar correlaciones
sns.pairplot(df_encoded[selected_features], diag_kind='kde')
plt.show()

#2.4 Estadisticas descriptivas 
print("\nDescriptive Statistics:")
print(df[['Gr Liv Area', 'SalePrice', 'Overall Qual']].describe())

#2.5 Histogramas para visualizar correlaciones
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

#2.6 Boxplot para visualizar correlaciones
# Sale Price
plt.figure(figsize = (10,8))
sns.boxplot(y = df ['SalePrice'])
plt.title('Boxplot of SalePrice')
plt.show()

stat, p = shapiro(df['SalePrice'])
print(f'P-value for SalePrice: {p}')

#Gr Liv Area 
plt.figure(figsize = (10,8))
sns.boxplot(y = df ['Gr Liv Area'])
plt.title('Boxplot of Gr Liv Area')
plt.show()

stat, p = shapiro(df['Gr Liv Area'])
print(f'P-value for Gr Liv Area: {p}') 

#Overall Qual 
plt.figure(figsize = (10,8))

sns.boxplot(y = df ['Overall Qual'])
plt.title('Boxplot of Overall Qual')
plt.show()

stat, p = shapiro(df['Overall Qual'])
print(f'P-value for Overall Qual: {p}') """

#2.7 Distribucion de los datos 
# Distribucion de datos
print(df['SalePrice'].skew())
print(df['Gr Liv Area'].skew())
print(df['Overall Qual'].skew())

#2.8 Busca valores menor o igual a 0
print()
print(df[df['SalePrice'] <= 0])
print()
print(df[df['Gr Liv Area'] <= 0])
print()
print(df[df['Overall Qual'] <= 0])

#2.9 Verifica valores nulos
print()
print(df[['SalePrice', 'Gr Liv Area', 'Overall Qual']].isnull().sum())

#2.10 Aplicamos logaritmicos
# Unimos las graficas antes de aplicar el logaritmo
"""fig,axes = plt.subplots(1, 3, figsize = (18, 5))

sns.histplot(df['SalePrice'], kde = True, bins = 30, ax = axes[0])
axes[0].set_title('Distribution of SalePrice before log')

sns.histplot(df['Gr Liv Area'], kde = True, bins = 30, ax = axes[1])
axes[1].set_title('Distribution of Gr Live Area before log')

sns.histplot(df['Overall Qual'], kde = True, bins = 30, ax = axes[2])
axes[2].set_title('Distribution of Overall Qual before log')

plt.show() """

# Se aplica los logaritmicos
df['SalePrice_log'] = np.log1p(df['SalePrice'])
df['Gr Liv Area_log'] = np.log1p(df['Gr Liv Area'])

"""fig, axes = plt.subplots(1, 2,  figsize = (12, 5))
sns.histplot(df['SalePrice_log'], kde = True, bins = 30, ax = axes[0])
axes[0].set_title("Distribution of SalePrice after log")

sns.histplot(df['Gr Liv Area_log'], kde = True, bins = 30, ax = axes[1])
axes[1].set_title("Distribution of Gr Liv Area after log")

plt.show()"""

#2.11 Normalizamos los datos
scaler = MinMaxScaler()
df[['SalePrice_log', 'Gr Liv Area_log']] = scaler.fit_transform(df[['SalePrice_log', 'Gr Liv Area_log']])
print(df[['SalePrice_log', 'Gr Liv Area_log']].head())
joblib.dump(scaler, '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/min_max_scaler.pkl')


#2.12 Visualizamos los datos normalizados 
"""fig, axes = plt.subplots(1, 3, figsize = (12, 5))
sns.histplot(df['SalePrice_log'], kde=True, bins = 30, ax = axes[0])
axes[0].set_title("SalePrice after MinMaxScaler")

sns.histplot(df['Gr Liv Area_log'], kde=True, bins = 30, ax = axes[1])
axes[1].set_title("Gr Liv Area after MinMaxScaler")

sns.histplot(df['Overall Qual'], kde=True, bins = 30, ax = axes[2])
axes[2].set_title("Overall Qual")

plt.show()"""
#---------------------------------------#

#--------- 4.Evaluation Metrics ---------#
def evalute_model(model, x_test, y_test, y_pred, model_name, feature):
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {model_name} for {feature}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print("-" * 50)
#---------------------------------------#

#------ 3.TRAINING MULTIPLE MODELS ------#
#----------------- 7.Ligth GBM ----------------- #
y = df['SalePrice_log']
x2 = df[['Overall Qual']]
x2_train, x2_test, y_train, y_test = train_test_split(x2, y, test_size = 0.2, random_state = 42)

gbm2 = lgb.LGBMRegressor(objective = 'regression', random_state = 42, verbosity = -1)
#gbm2.fit(x2_train, y_train)
gbm2.fit(x2_train[['Overall Qual']], y_train)
y_pred_rr2 = gbm2.predict(x2_test)

# Guardar las columnas del conjunto de entrenamiento
#joblib.dump(x2_train.columns.tolist(), 'feature_names.pkl')
#Evaluar Modelos
print()
evalute_model(gbm2, x2_test, y_test, y_pred_rr2, "LightGBM Regressor", "Overall Qual" )
#----------------------------------------#

#----------------- 5. FEATURE ENGINEERING MANUAL ----------------- #

#----------------- 1. Optuna ----------------- #
def objetive(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30)
    }

    model = lgb.LGBMRegressor(**param)
    model.fit(x2_train, y_train)
    preds = model.predict(x2_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return rmse

study = optuna.create_study(direction = 'minimize')
study.optimize(objetive, n_trials = 50)

print()
print("Best Parameters:")
print(study.best_params)
print()

#Evaluar el modelo
best_params_optuna = study.best_params
best_optuna = lgb.LGBMRegressor(n_estimators = best_params_optuna['n_estimators'], max_depth = best_params_optuna['max_depth'], learning_rate = best_params_optuna['learning_rate'], min_child_samples = best_params_optuna['min_child_samples'], random_state = 42)
best_optuna.fit(x2_train, y_train)

y_pred_optuna = best_optuna.predict(x2_test)

mse_optuna = mean_squared_error(y_test, y_pred_optuna)
rmse_optuna = np.sqrt(mse_optuna)
r2_optuna = r2_score(y_test, y_pred_optuna)

print()
print("Evaluation of optimization model with Optuna:")
print(f"MSE: {mse_optuna: .4f}")
print(f"RMSE: {rmse_optuna: .4f}")
print(f"R2: {r2_optuna: .4f}")
print()

#----------------- 2. Early Stopping -----------------#
gbm = lgb.LGBMRegressor(objective = 'regression', random_state = 42, n_estimators=1000) 
gbm.fit(x2_train, y_train, eval_set = [(x2_test, y_test)], eval_metric = 'rmse', callbacks = [early_stopping(stopping_rounds = 50), log_evaluation(0)])

y_pred_early = gbm.predict(x2_test)

mse_early = mean_squared_error(y_test, y_pred_early)
rmse_early = np.sqrt(mse_early)
r2_early = r2_score(y_test, y_pred_early)

print()
print("Evaluation of model with Early Stopping:")
print(f"MSE: {mse_early: .4f}")
print(f"RMSE: {rmse_early: .4f}")
print(f"r2: {r2_early: .4f}")
print()
#------------------------------------------------------#


#----------------- 6.FEATURE ENGINEERING MANUAL -----------------#
"""
#6.1 Crear nuevas variables
# 6.1.1 Crear Relaciones entre variables 
df['TotalBathrooms'] = (df['Full Bath'] + (0.5 * df['Half Bath']) + df['Bsmt Full Bath'] + (0.5 * df['Bsmt Half Bath']))

df['TotalSF'] = df['Total Bsmt SF'] + df['1st Flr SF'] + df['2nd Flr SF']

df['GarageRatio'] = df ['Garage Area'] / df['Gr Liv Area']

# 6.1.2 Crear variable de antiguedad 
df['HouseAge'] = df['Yr Sold'] - df['Year Built']
df['RemodAge'] = df['Yr Sold'] - df['Year Remod/Add']
df['GarageAge'] = df['Yr Sold'] - df['Garage Yr Blt']

# 6.1.3 Combinar Informacion de la Zona
neighborhood_map = df.groupby('Neighborhood')['SalePrice'].mean().sort_values()
df['NeighborhoodValue'] = df['Neighborhood'].map(neighborhood_map)

# 6.1.4 Bandas de calidad o tamaño 
df['Qual_Band'] = pd.cut(df['Overall Qual'], bins = [0, 4, 6, 8, 10], labels = ['Baja', 'Media', 'Alta', 'Premium'])
df['Size_Band'] = pd.cut(df['Gr Liv Area'], bins = 4, labels = ['Pequeñas', 'Mediana', 'Grande', 'XL'])

# 6.1.5 Bandas Temporales 
df['SaleDecade'] = pd.cut(df['Yr Sold'], bins = [2005, 2007, 2009, 2011], labels = ['2006-07', '2008-09', '2010-11'])

# Ver las primeras filas de tus nuevas features
print(df[['TotalBathrooms', 'TotalSF', 'GarageRatio', 'HouseAge', 'RemodAge', 'GarageAge',
          'NeighborhoodValue', 'Qual_Band', 'Size_Band', 'SaleDecade']].head(10))

#6.2 Mini Exploratory Data Analysis
#6.2.1 Matriz de correlacion 
#Variables especificas que quieres analizar
features = ['SalePrice', 'Overall Qual', 'TotalBathrooms', 'TotalSF', 'GarageRatio', 'HouseAge', 'RemodAge', 'GarageAge', 'NeighborhoodValue']# 'Qual_Band', 'Size_Band', 'SaleDecade']
#Seleccionamos solo esas columnas 
subset = df[features]
#Calculamos la correlacion 
corr_matrix = subset.corr()
#Hacemos el heatmap 
plt.figure(figsize = (12,8))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm', fmt = ".2f")
plt.title('Correlation Matrix of Selected Features')
plt.show()
"""
#--------------------------------------------------------#

#----------------- 7. DATA AUGMENTATION -----------------#
#--------------------------------------------------------#

#----------------- 8. FINAL MODEL EVALUATION -----------------#

#Light GBM Base
print("Final Model Evaluation - LightGBM Base")
#Predecir en el conjunto de prueba 
y_pred = gbm2.predict(x2_test)

#9.Conversiones 
# Al predecir con los modelos, deshacer la transformación logarítmica para obtener la predicción original
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred)  # Exponencial inversa de la predicción

#calcular metricas de evaluacion 
mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

#Mostrar Resultados 
print()
print("Final Model Evaluation ")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (NSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

#Light GBM + Optuna
print("Final Model Evaluation - LightGBM + Optuna")
#Predecir en el conjunto de prueba 
y_pred_optuna_final = best_optuna.predict(x2_test)
#9.Conversiones 
# Al predecir con los modelos, deshacer la transformación logarítmica para obtener la predicción original
y_pred_optuna_original = np.expm1(y_pred_optuna_final)  # Exponencial inversa de la predicción
y_test_optuna_original = np.expm1(y_test)

#calcular metricas de evaluacion 
mae2 = mean_absolute_error(y_test_optuna_original, y_pred_optuna_original)
mse2 = mean_squared_error(y_test_optuna_original, y_pred_optuna_original)
rmse2 = np.sqrt(mse2)
r2 = r2_score(y_test_optuna_original, y_pred_optuna_original)

#Mostrar Resultados
print() 
print("Final Model Evaluation ")
print(f"Mean Absolute Error (MAE): {mae2:.4f}")
print(f"Mean Squared Error (NSE): {mse2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse2:.4f}")
print(f"R2 Score: {r2:.4f}")

#Light GBM + Early Stopping
print("Final Model Evaluation - LightGBM + Early Stopping")
#Predecir en el conjunto de prueba 
y_pred_early_final = gbm.predict(x2_test)

#9.Conversiones 
# Al predecir con los modelos, deshacer la transformación logarítmica para obtener la predicción original
y_pred_early_original = np.expm1(y_pred_early_final)  # Exponencial inversa de la predicción
y_test_early_original = np.expm1(y_test)

#calcular metricas de evaluacion 
mae3 = mean_absolute_error(y_test_early_original, y_pred_early_original)
mse3 = mean_squared_error(y_test_early_original, y_pred_early_original)
rmse3 = np.sqrt(mse3)
r2 = r2_score(y_test_early_original, y_pred_early_original)

#Mostrar Resultados 
print()
print("Final Model Evaluation ")
print(f"Mean Absolute Error (MAE): {mae3:.4f}")
print(f"Mean Squared Error (NSE): {mse3:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse3:.4f}")
print(f"R2 Score: {r2:.4f}")
#--------------------------------------------------------#

#9. Export Model 
#Guarda el modelo con Optuna (Mejor modelo)
joblib.dump(gbm2, '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/lightgbm_model.pkl')
joblib.dump(list(x2_train.columns), '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/feature_names.pkl')

joblib.dump(best_optuna, '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/lightgbm_optuna_model.pkl')
joblib.dump(gbm, '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/lightgbm_early_model.pkl')
