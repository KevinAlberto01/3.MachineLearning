import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Loading Cleaned Data
print("\n=== 1. Loading Cleaned Data ===")
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv'
df = pd.read_csv(file_path)
print(f"Dataset loaded with shape: {df.shape}")

# Separar features y target
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Detectar columnas categóricas
categorical_columns = X.select_dtypes(include=['object']).columns
print(f"\nDetected Categorical Columns: {list(categorical_columns)}")

# Aplicar One-Hot Encoding
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# División train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Modelos
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'KNeighborsRegressor': KNeighborsRegressor()
}

# DataFrame para guardar resultados
results = []

# Función de evaluación
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n📊 Model: {model.__class__.__name__}")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Train R2: {train_r2:.2f}")
    print(f"Test R2: {test_r2:.2f}")
    print("-" * 50)

    results.append({
        'Model': model.__class__.__name__,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2
    })

# Evaluar todos los modelos
for name, model in models.items():
    evaluate_model(model, X_train, X_test, y_train, y_test)

# Convertir resultados a DataFrame
df_results = pd.DataFrame(results)

# Ordenar por el orden de los modelos
df_results['Model'] = pd.Categorical(df_results['Model'], categories=models.keys(), ordered=True)
df_results = df_results.sort_values('Model')

# 6. Comparación gráfica
print("\n=== 6. Comparación Gráfica ===")

plt.figure(figsize=(14, 6))

x = np.arange(len(df_results['Model']))
width = 0.35  # Ancho de las barras

# Gráfica de RMSE
plt.subplot(1, 2, 1)
plt.bar(x - width/2, df_results['Train RMSE'], width=width, label='Train RMSE')
plt.bar(x + width/2, df_results['Test RMSE'], width=width, label='Test RMSE')
plt.xticks(x, df_results['Model'], rotation=45)
plt.ylabel('RMSE')
plt.title('Comparation of RMSE')
plt.legend()

# Gráfica de R²
plt.subplot(1, 2, 2)
plt.bar(x - width/2, df_results['Train R²'], width=width, label='Train R²')
plt.bar(x + width/2, df_results['Test R²'], width=width, label='Test R²')
plt.xticks(x, df_results['Model'], rotation=45)
plt.ylabel('R²')
plt.title('Comparation de R²')
plt.legend()

plt.tight_layout()
plt.show()
