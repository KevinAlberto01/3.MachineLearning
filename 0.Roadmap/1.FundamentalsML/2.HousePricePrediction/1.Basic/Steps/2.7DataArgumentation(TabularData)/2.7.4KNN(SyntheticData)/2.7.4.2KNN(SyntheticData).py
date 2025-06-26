#1.IMPORTING LIBRARIES
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

#2.FUNCTION TO GENERATE SYNTHETIC DATA WITH KNN
def generate_knn_synthetic_data(X, y, k_neighbors=5, synthetic_points=500):
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    data = np.hstack((X, y))

    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(data)

    synthetic_data = []

    for _ in range(synthetic_points):
        idx = np.random.choice(len(X))  #2.1 Choose a random real point.
        distances, neighbors_idx = nbrs.kneighbors([data[idx]])
        neighbors = data[neighbors_idx[0]]

        #2.2 Choose a random neighbor (different from the original point).
        neighbor = neighbors[np.random.choice(range(1, len(neighbors)))]

        #2.3 Interpolate synthetic point.
        alpha = np.random.rand()
        synthetic_point = alpha * data[idx] + (1 - alpha) * neighbor
        synthetic_data.append(synthetic_point)

    synthetic_data = np.array(synthetic_data)
    X_synthetic = synthetic_data[:, :-1]
    y_synthetic = synthetic_data[:, -1]

    #2.4 Concatenate real data with synthetic data.
    X_augmented = np.vstack([X, X_synthetic])
    y_augmented = np.concatenate([y.flatten(), y_synthetic])

    return X_augmented, y_augmented


#3.LOADING AND PREPARING THE DATA
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Jittering.csv')

#3.1 Feature Engineering
df['TotalBathrooms'] = df['full_bath'] + df['half_bath'] * 0.5
df['HouseAge'] = 2025 - df['year_built']
df['PricePerSF'] = df['saleprice'] / df['gr_liv_area']

#4.PREPARATION FOR MODELING
X = pd.get_dummies(df.drop(columns=['saleprice']), drop_first=True)
y = df['saleprice']

#4.1 Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

#4.2Generate synthetic data.
X_augmented, y_augmented = generate_knn_synthetic_data(X_scaled, y, k_neighbors=5, synthetic_points=500)

#5.SPLITTING INTO TRAINING AND TESTING SETS
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)

#6.TRAINING MODELS
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

#6.1 Results
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

#7.SAVING AND VISUALIZING THE RESULTS
results_df = pd.DataFrame(results)
print(results_df)

#7.1 Save results to CSV.
results_df.to_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/KNN(SyntheticDataJittering).csv', index=False)

plt.figure(figsize=(12, 6))

#7.2 Test RMSE plot.
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Test RMSE', data=results_df, hue='Model', palette='viridis', legend=False)
plt.title('Comparación de Test RMSE por Modelo')
plt.xticks(rotation=45)

#7.3 Test R² plot.
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='Test R²', data=results_df, hue='Model', palette='plasma', legend=False)
plt.title('Comparación de Test R² por Modelo')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
