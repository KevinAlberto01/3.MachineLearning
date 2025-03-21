import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score

# Función para generar datos sintéticos con KNN
def generate_knn_synthetic_data(X, y, k_neighbors=5, synthetic_points=500):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    data = np.hstack((X, y))

    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(data)

    synthetic_data = []

    for _ in range(synthetic_points):
        idx = np.random.choice(len(X))  # Elegir punto aleatorio real
        distances, neighbors_idx = nbrs.kneighbors([data[idx]])
        neighbors = data[neighbors_idx[0]]

        # Elegir un vecino aleatorio (diferente al punto original)
        neighbor = neighbors[np.random.choice(range(1, len(neighbors)))]

        # Interpolar punto sintético
        alpha = np.random.rand()
        synthetic_point = alpha * data[idx] + (1 - alpha) * neighbor
        synthetic_data.append(synthetic_point)

    synthetic_data = np.array(synthetic_data)
    X_synthetic = synthetic_data[:, :-1]
    y_synthetic = synthetic_data[:, -1]

    # Concatenar datos reales con sintéticos
    X_augmented = np.vstack([X, X_synthetic])
    y_augmented = np.concatenate([y.flatten(), y_synthetic])

    return X_augmented, y_augmented


# Cargar dataset
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Jittering.csv')

# Feature Engineering
df['TotalBathrooms'] = df['full_bath'] + df['half_bath'] * 0.5
df['HouseAge'] = 2025 - df['year_built']
df['PricePerSF'] = df['saleprice'] / df['gr_liv_area']

# Variables y target
X = pd.get_dummies(df.drop(columns=['saleprice']), drop_first=True)
y = df['saleprice']

# Escalamos
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Generar datos sintéticos
X_augmented, y_augmented = generate_knn_synthetic_data(X_scaled, y, k_neighbors=5, synthetic_points=500)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

# Modelos
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

# Resultados
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        'Model': name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2
    })

# Convertir resultados a DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# Guardar resultados en CSV
results_df.to_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/KNN(SyntheticDataJittering).csv', index=False)

plt.figure(figsize=(12, 6))

# Gráfico de Test RMSE
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Test RMSE', data=results_df, hue='Model', palette='viridis', legend=False)
plt.title('Comparación de Test RMSE por Modelo')
plt.xticks(rotation=45)

# Gráfico de Test R²
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Test R²', data=results_df, hue='Model', palette='plasma', legend=False)
plt.title('Comparación de Test R² por Modelo')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
