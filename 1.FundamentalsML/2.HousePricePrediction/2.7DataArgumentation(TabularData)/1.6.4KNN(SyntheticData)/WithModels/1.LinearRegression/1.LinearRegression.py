import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

# Función KNN Synthetic Data (la misma de arriba, la puedes importar también)
def generate_knn_synthetic_data(X, y, k_neighbors=5, synthetic_points=500):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    data = np.hstack((X, y))
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(data)
    synthetic_data = []
    for _ in range(synthetic_points):
        idx = np.random.choice(len(X))
        distances, neighbors_idx = nbrs.kneighbors([data[idx]])
        neighbors = data[neighbors_idx[0]]
        neighbor = neighbors[np.random.choice(range(1, len(neighbors)))]
        alpha = np.random.rand()
        synthetic_point = alpha * data[idx] + (1 - alpha) * neighbor
        synthetic_data.append(synthetic_point)
    synthetic_data = np.array(synthetic_data)
    X_synthetic = synthetic_data[:, :-1]
    y_synthetic = synthetic_data[:, -1]
    X_augmented = np.vstack([X, X_synthetic])
    y_augmented = np.concatenate([y.flatten(), y_synthetic])
    return X_augmented, y_augmented

# Load data
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

# Feature Engineering
df['TotalBathrooms'] = df['Full Bath'] + df['Half Bath'] * 0.5
df['HouseAge'] = 2025 - df['Year Built']
df['PricePerSF'] = df['SalePrice'] / df['Gr Liv Area']

X = pd.get_dummies(df.drop(columns=['SalePrice']), drop_first=True)
y = df['SalePrice']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Generar datos sintéticos
X_augmented, y_augmented = generate_knn_synthetic_data(X_scaled, y, k_neighbors=5, synthetic_points=500)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

# Modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Métricas
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Linear Regression (KNN Synthetic Data) - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}')
print(f'Linear Regression (KNN Synthetic Data) - Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}')
