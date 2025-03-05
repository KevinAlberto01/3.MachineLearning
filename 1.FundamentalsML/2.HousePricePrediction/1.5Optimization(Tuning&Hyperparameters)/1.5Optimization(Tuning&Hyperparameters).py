import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

# Mostrar resumen de SalePrice
print("Resumen de SalePrice:")
print(df['SalePrice'].describe())

# Boxplot de SalePrice (sin seaborn)
plt.figure(figsize=(8, 4))
plt.boxplot(df['SalePrice'], vert=False)
plt.title("Distribución de SalePrice")
plt.xlabel("SalePrice")
plt.show()

# Preparar datos (get_dummies)
X = pd.get_dummies(df.drop(columns=['SalePrice']), drop_first=True)
y = df['SalePrice']

# Revisar nulos y valores negativos
print("Nulos en X:", X.isnull().sum().sum())
print("Valores negativos en X:", (X < 0).sum().sum())

# Determinar tipo de escalado
type_scaling = 'StandardScaler' if X.max().max() < 1e3 else 'RobustScaler'
print(f"Escalado recomendado: {type_scaling}")

scaler = StandardScaler() if type_scaling == 'StandardScaler' else RobustScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar log1p a SalePrice (opcional)
log_transform = input("¿Aplicar log1p a SalePrice? (s/n): ")
if log_transform.lower() == 's':
    y = np.log1p(y)
    print("Transformación log aplicada a SalePrice.")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelos
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

# Inicializar lista para guardar resultados
results_list = []

# Entrenar y evaluar cada modelo
for name, model in models.items():
    model.fit(X_train, y_train)

    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calcular métricas
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Guardar resultados en lista
    results_list.append({
        'Model': name,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_R2': train_r2,
        'Test_R2': test_r2
    })

# Convertir lista a DataFrame
results = pd.DataFrame(results_list)

# Mostrar resultados
print("\nComparación de Modelos:")
print(results)

# Guardar resultados
results.to_csv('model_comparison_results.csv', index=False)

# Gráficas de comparación
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

bar_width = 0.35
index = np.arange(len(results['Model']))

# Gráfico de RMSE
axs[0].bar(index - bar_width/2, results['Train_RMSE'], bar_width, label='Train RMSE')
axs[0].bar(index + bar_width/2, results['Test_RMSE'], bar_width, label='Test RMSE')
axs[0].set_xticks(index)
axs[0].set_xticklabels(results['Model'], rotation=45, ha='right')
axs[0].set_title('Comparación de RMSE')
axs[0].set_xlabel('Modelo')
axs[0].set_ylabel('RMSE')
axs[0].legend()

# Gráfico de R²
axs[1].bar(index - bar_width/2, results['Train_R2'], bar_width, label='Train R²')
axs[1].bar(index + bar_width/2, results['Test_R2'], bar_width, label='Test R²')
axs[1].set_xticks(index)
axs[1].set_xticklabels(results['Model'], rotation=45, ha='right')
axs[1].set_title('Comparación de R²')
axs[1].set_xlabel('Modelo')
axs[1].set_ylabel('R²')
axs[1].legend()

plt.tight_layout()
plt.show()
