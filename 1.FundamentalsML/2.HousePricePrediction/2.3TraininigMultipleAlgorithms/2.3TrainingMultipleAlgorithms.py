import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# 1. Loading Cleaned Data
print("\n=== 1. Loading Cleaned Data ===")
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv'
df = pd.read_csv(file_path)
print(f"Dataset loaded with shape: {df.shape}")

# Separar features y target
X = df.drop(columns=['saleprice'])
y = df['saleprice']

# Detectar columnas categóricas
categorical_columns = X.select_dtypes(include=['object']).columns
print(f"\nDetected Categorical Columns: {list(categorical_columns)}")
print()

# Aplicar One-Hot Encoding
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Verificación de la forma de X después de One-Hot Encoding
print(f"Shape of X after One-Hot Encoding: {X.shape}")
print(X.head())
print()

# División train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Normalización de las características ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Grid Search para DecisionTreeRegressor ---
def get_best_decision_tree(X_train, y_train):
    param_grid = {'max_depth': [5, 10, 15, 20, 25, None]}
    grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print(f"\n✅ Best max_depth for DecisionTreeRegressor: {grid_search.best_params_['max_depth']}")
    return DecisionTreeRegressor(random_state=42, max_depth=grid_search.best_params_['max_depth'])

# Llamamos a la función de GridSearch para obtener el mejor modelo de árbol de decisión
best_decision_tree = get_best_decision_tree(X_train, y_train)

# Modelos
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': best_decision_tree,  # Optimizado con el mejor max_depth
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
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_ev = explained_variance_score(y_train, y_train_pred)
    test_ev = explained_variance_score(y_test, y_test_pred)

    print(f"\n📊 Model: {model.__class__.__name__}")
    print(f"Train RMSE: {train_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    print(f"Train R²: {train_r2:.2f}")
    print(f"Test R²: {test_r2:.2f}")
    print(f"Train MAE: {train_mae:.2f}")
    print(f"Test MAE: {test_mae:.2f}")
    print(f"Train Explained Variance: {train_ev:.2f}")
    print(f"Test Explained Variance: {test_ev:.2f}")
    print("-" * 50)

    results.append({
        'Model': model.__class__.__name__,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train Explained Variance': train_ev,
        'Test Explained Variance': test_ev
    })

# Evaluar todos los modelos
for name, model in models.items():
    evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)

# Convertir resultados a DataFrame
df_results = pd.DataFrame(results)

# Ordenar por el orden de los modelos
df_results['Model'] = pd.Categorical(df_results['Model'], categories=models.keys(), ordered=True)
df_results = df_results.sort_values('Model')

# Guardar los resultados
df_results.to_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/df_results.csv', index=False)
