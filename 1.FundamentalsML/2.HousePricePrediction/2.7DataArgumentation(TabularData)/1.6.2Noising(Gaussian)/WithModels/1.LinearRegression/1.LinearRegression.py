import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Gaussian Noise
def add_gaussian_noise(X, noise_factor=0.01):
    X_noisy = X.copy()
    for col in X_noisy.columns:
        noise = np.random.normal(0, noise_factor * X_noisy[col].std(), X_noisy.shape[0])
        X_noisy[col] += noise
    return X_noisy

X_augmented = add_gaussian_noise(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Linear Regression - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}')
print(f'Linear Regression - Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}')
