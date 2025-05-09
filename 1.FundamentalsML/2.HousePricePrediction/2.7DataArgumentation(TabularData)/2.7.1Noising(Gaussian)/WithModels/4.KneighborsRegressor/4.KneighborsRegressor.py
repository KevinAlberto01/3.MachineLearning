#1.IMPORT LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_squared_error, r2_score

#2.DATASET LOADING
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

#3.FEATURE ENGINEERING
df['TotalBathrooms'] = df['full_bath'] + df['half_bath'] * 0.5
df['HouseAge'] = 2025 - df['year_built']
df['PricePerSF'] = df['saleprice'] / df['gr_liv_area']

#4.DATA PREPARATION
X = pd.get_dummies(df.drop(columns=['saleprice']), drop_first=True)
y = df['saleprice']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

#5.DATA AUGMENTATION (GAUSSIAN NOISE)
def add_gaussian_noise(X, noise_factor=0.01):
    X_noisy = X.copy()
    for col in X_noisy.columns:
        noise = np.random.normal(0, noise_factor * X_noisy[col].std(), X_noisy.shape[0])
        X_noisy[col] += noise
    return X_noisy

X_augmented = add_gaussian_noise(X_scaled)

#6.DIVISION INTO TRAINING AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y, test_size=0.2, random_state=42)

#7.KNN MODEL TRAINING
model = KNeighborsRegressor()
model.fit(X_train, y_train)

#8.MODEL PREDICTION AND EVALUATION
#8.1 Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#8.2 Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'K-Nearest Neighbors Regressor - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}')
print(f'K-Nearest Neighbors Regressor - Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}')
