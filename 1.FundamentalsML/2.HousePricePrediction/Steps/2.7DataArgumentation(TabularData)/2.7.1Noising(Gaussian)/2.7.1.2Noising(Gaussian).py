#1.IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

#2.LOAD DATA
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Jittering.csv')
print(df.columns)

#3.FEATURE ENGINEERING
df['TotalBathrooms'] = df['full_bath'] + df['half_bath'] * 0.5
df['HouseAge'] = 2025 - df['year_built']
df['PricePerSF'] = df['saleprice'] / df['gr_liv_area']

#4.DATA PREPARATION FOR THE MODEL 
X = pd.get_dummies(df.drop(columns=['saleprice']), drop_first=True)
y = df['saleprice']

#5.DATA NORMALIZATION
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

#6.APPLYING GAUSSIAN NOISE (DATA AUGMENTATION)
def add_gaussian_noise(X, noise_factor=0.01):
    X_noisy = X.copy()
    for col in X_noisy.columns:
        noise = np.random.normal(0, noise_factor * X_noisy[col].std(), X_noisy.shape[0])
        X_noisy[col] += noise
    return X_noisy

X_augmented = add_gaussian_noise(X_scaled)

#7.DIVISION INTO TRAINING AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y, test_size=0.2, random_state=42)

#8.DEFINE MACHINE LEARNING MODELS 
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

#9.TRAINING AND EVALUATION OF MODELS
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

#10.SAVING RESULTS
#10.1 Results DataFrame
results_df = pd.DataFrame(results)
#10.2 Save results
results_df.to_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/model_comparison_with_gaussian(Jittering).csv', index=False)
#10.3 Print Results
print(results_df)

#11.METRICS VISUALIZATION
#11.1 Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#11.2RMSE Plot
axes[0].bar(results_df['Model'], results_df['Train RMSE'], label='Train RMSE', alpha=0.6)
axes[0].bar(results_df['Model'], results_df['Test RMSE'], label='Test RMSE', alpha=0.6)
axes[0].set_title('Train vs Test RMSE')
axes[0].legend()

#11.3 R² Plot
axes[1].bar(results_df['Model'], results_df['Train R²'], label='Train R²', alpha=0.6)
axes[1].bar(results_df['Model'], results_df['Test R²'], label='Test R²', alpha=0.6)
axes[1].set_title('Train vs Test R²')
axes[1].legend()

plt.tight_layout()
plt.show()
