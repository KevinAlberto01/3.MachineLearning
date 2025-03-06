import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load original dataset
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')
#print(df.columns)

# Feature Engineering Manual (Basic Examples)
df['TotalBathrooms'] = df['Full Bath'] + df['Half Bath'] * 0.5
df['HouseAge'] = 2025 - df['Year Built']
df['PricePerSF'] = df['SalePrice'] / df['Gr Liv Area']

# Preprocessing
X = pd.get_dummies(df.drop(columns=['SalePrice']), drop_first=True)
y = df['SalePrice']

# Scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

# Results
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

# Save results
results_df = pd.DataFrame(results)
print(results_df)

results_df.to_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.6DataArgumentation(TabularData)/1.6.1FeatureEngineeringManual/results_feature_engineering.csv', index=False)
