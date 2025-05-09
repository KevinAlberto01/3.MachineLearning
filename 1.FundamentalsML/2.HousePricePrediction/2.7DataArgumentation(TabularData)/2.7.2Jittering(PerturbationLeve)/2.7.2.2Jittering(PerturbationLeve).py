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

#2.DEFINITION OF FUNCTION
def add_jitter(X, perturbation_level=0.01):
    X_jittered = X.copy()
    for col in X_jittered.columns:
        perturbation = np.random.uniform(-perturbation_level, perturbation_level, X_jittered.shape[0]) * X_jittered[col].std()
        X_jittered[col] += perturbation
    return X_jittered

#3.LOADING DATASET
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Jittering.csv')

#4.FEATURE ENGINEERING
df['TotalBathrooms'] = df['full_bath'] + df['half_bath'] * 0.5
df['HouseAge'] = 2025 - df['year_built']
df['PricePerSF'] = df['saleprice'] / df['gr_liv_area']

#5.DATA PREPARATION
X = pd.get_dummies(df.drop(columns=['saleprice']), drop_first=True)
y = df['saleprice']

#5.1 Model Preparation
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

#5.2 Apply Jittering
X_augmented = add_jitter(X_scaled)
 
#6.DIVISION INTO TRAINING AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y, test_size=0.2, random_state=42)

#7.TRAINING AND EVALUATION OF MODELS
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

#7.1 Results collection
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

#8.SAVING AND VIEWING RESULTS
#8.1 Results DataFrame
results_df = pd.DataFrame(results)

#8.2 Save results
results_df.to_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/model_comparison_with_jittering.csv', index=False)

#8.3 Print
print(results_df)

#8.4 Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#8.5 RMSE Plot
axes[0].bar(results_df['Model'], results_df['Train RMSE'], label='Train RMSE', alpha=0.6)
axes[0].bar(results_df['Model'], results_df['Test RMSE'], label='Test RMSE', alpha=0.6)
axes[0].set_title('Train vs Test RMSE (Jittering)')
axes[0].legend()

#8.6 R² Plot
axes[1].bar(results_df['Model'], results_df['Train R²'], label='Train R²', alpha=0.6)
axes[1].bar(results_df['Model'], results_df['Test R²'], label='Test R²', alpha=0.6)
axes[1].set_title('Train vs Test R² (Jittering)')
axes[1].legend()

plt.tight_layout()
plt.show()
