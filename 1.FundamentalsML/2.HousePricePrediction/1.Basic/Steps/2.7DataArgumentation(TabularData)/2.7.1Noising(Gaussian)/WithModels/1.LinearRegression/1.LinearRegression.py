#1.IMPORTING LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
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

#5.NORMALIZATION WITH ROBUSTSCALER
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

#6.JITTERING(GAUSSIAN NOISE FOR DATA AUGMENTATION)
def add_gaussian_noise(X, noise_factor=0.01):
    X_noisy = X.copy()
    for col in X_noisy.columns:
        noise = np.random.normal(0, noise_factor * X_noisy[col].std(), X_noisy.shape[0])
        X_noisy[col] += noise
    return X_noisy

X_augmented = add_gaussian_noise(X_scaled)

#7.TRAINING AND TEST SET PARTITIONING
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y, test_size=0.2, random_state=42)

#8.LINEAR REGRESSION MODEL TRAINING
model = LinearRegression()
model.fit(X_train, y_train)

#9.PREDICTIONS
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#10.MODEL EVALUATION
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Linear Regression - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}')
print(f'Linear Regression - Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}')
