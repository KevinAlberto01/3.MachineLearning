import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

# Show summary of SalePrice
print("Summary of SalePrice:")
print(df['SalePrice'].describe())

# Boxplot of SalePrice (without seaborn)
plt.figure(figsize=(8, 4))
plt.boxplot(df['SalePrice'], vert=False)
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.show()

# Prepare data (get_dummies)
X = pd.get_dummies(df.drop(columns=['SalePrice']), drop_first=True)
y = df['SalePrice']

# Check for nulls and negative values
print("Null in X:", X.isnull().sum().sum())
print("Negative Value X:", (X < 0).sum().sum())

# Determine type of scaling
type_scaling = 'StandardScaler' if X.max().max() < 1e3 else 'RobustScaler'
print(f"Recommended escalation: {type_scaling}")

scaler = StandardScaler() if type_scaling == 'StandardScaler' else RobustScaler()
X_scaled = scaler.fit_transform(X)

# Apply log1p to SalePrice (optional)
log_transform = input("Apply log1p to SalePrice? (y/n): ")
if log_transform.lower() == 's':
    y = np.log1p(y)
    print("Log transformation applied to SalePrice.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

# Initialize list to store results
results_list = []

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Save results in list
    results_list.append({
        'Model': name,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_R2': train_r2,
        'Test_R2': test_r2
    })

# Convert list to DataFrame
results = pd.DataFrame(results_list)

# Show results
print("\nComparation of Models:")
print(results)

# Save results
results.to_csv('3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.5Optimization(Tuning&Hyperparameters)/model_comparison_results.csv', index=False)

# Comparison charts
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

bar_width = 0.35
index = np.arange(len(results['Model']))

# RMSE Chart
axs[0].bar(index - bar_width/2, results['Train_RMSE'], bar_width, label='Train RMSE')
axs[0].bar(index + bar_width/2, results['Test_RMSE'], bar_width, label='Test RMSE')
axs[0].set_xticks(index)
axs[0].set_xticklabels(results['Model'], rotation=45, ha='right')
axs[0].set_title('Comparation of RMSE')
axs[0].set_xlabel('Model')
axs[0].set_ylabel('RMSE')
axs[0].legend()

# R² Graph
axs[1].bar(index - bar_width/2, results['Train_R2'], bar_width, label='Train R²')
axs[1].bar(index + bar_width/2, results['Test_R2'], bar_width, label='Test R²')
axs[1].set_xticks(index)
axs[1].set_xticklabels(results['Model'], rotation=45, ha='right')
axs[1].set_title('Comparation of R²')
axs[1].set_xlabel('Model')
axs[1].set_ylabel('R²')
axs[1].legend()

plt.tight_layout()
plt.show()
