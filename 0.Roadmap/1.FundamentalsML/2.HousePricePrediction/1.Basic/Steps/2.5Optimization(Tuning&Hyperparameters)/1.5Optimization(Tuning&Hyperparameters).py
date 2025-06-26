#1.IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#2.LOAD DATA AND VISUALIZATION
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

#2.1 Show summary of SalePrice
print("Summary of SalePrice:")
print(df['saleprice'].describe())

#2.2 Boxplot of SalePrice (without seaborn)
plt.figure(figsize=(8, 4))
plt.boxplot(df['saleprice'], vert=False)
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.show()

#3.PREPROCESSING DATA
X = pd.get_dummies(df.drop(columns=['saleprice']), drop_first=True)
y = df['saleprice']

#3.1 Check for nulls and negative values
print("Null in X:", X.isnull().sum().sum())
print("Negative Value X:", (X < 0).sum().sum())

#4.SELECTING THE SCALING TYPE (NORMALIZATION)
type_scaling = 'StandardScaler' if X.max().max() < 1e3 else 'RobustScaler'
print(f"Recommended escalation: {type_scaling}")

scaler = StandardScaler() if type_scaling == 'StandardScaler' else RobustScaler()
X_scaled = scaler.fit_transform(X)

#5.LOGARITHMIC TRANSFORMATION OF THE TARGET VARIABLE
log_transform = input("Apply log1p to SalePrice? (y/n): ")
if log_transform.lower() == 's':
    y = np.log1p(y)
    print("Log transformation applied to SalePrice.")

#6.DIVISION OF DATA INTO TRAINING AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#7.Training and evaluation of models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

#7.1 Initialize list to store results
results_list = []

#7.2 Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)

    #7.3 Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    #7.4 Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    #7.5 Save results in list
    results_list.append({
        'Model': name,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Train_MAE': train_mae,
        'Test_MAE': test_mae
    })

#7.6 Convert list to DataFrame
results = pd.DataFrame(results_list)

#7.7 Show results
print("\nComparation of Models:")
print(results)

#7.8 Save results
results.to_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning&Hyperparameters)/AmesHousing_cleaned.csv', index=False)

#8.GRAPHICAL COMPARISON OF RESULTS 
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

bar_width = 0.35
index = np.arange(len(results['Model']))

#8.1 RMSE Chart
axs[0].bar(index - bar_width/2, results['Train_RMSE'], bar_width, label='Train RMSE')
axs[0].bar(index + bar_width/2, results['Test_RMSE'], bar_width, label='Test RMSE')
axs[0].set_xticks(index)
axs[0].set_xticklabels(results['Model'], rotation=45, ha='right')
axs[0].set_title('Comparation of RMSE')
axs[0].set_xlabel('Model')
axs[0].set_ylabel('RMSE')
axs[0].legend()

#8.2 R² Graph
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

#9.HYPERPARAMETER OPTIMIZATION WITH GRIDSEARCH CV FOR RANDOM FOREST
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)  # verbose=1
grid_search.fit(X_train, y_train)

print()
print("\nBest Parameters for Random Forest:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_

#10.EVALUATION OF THE BEST RANDOM FOREST MODEL
y_train_pred_best_rf = best_rf_model.predict(X_train)
y_test_pred_best_rf = best_rf_model.predict(X_test)

train_rmse_best_rf = np.sqrt(mean_squared_error(y_train, y_train_pred_best_rf))
test_rmse_best_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_best_rf))
train_r2_best_rf = r2_score(y_train, y_train_pred_best_rf)
test_r2_best_rf = r2_score(y_test, y_test_pred_best_rf)
train_mae_best_rf = mean_absolute_error(y_train, y_train_pred_best_rf)
test_mae_best_rf = mean_absolute_error(y_test, y_test_pred_best_rf)

print("\nEvaluation of Best Random Forest Model:")
print(f"Train RMSE: {train_rmse_best_rf}")
print(f"Test RMSE: {test_rmse_best_rf}")
print(f"Train R²: {train_r2_best_rf}")
print(f"Test R²: {test_r2_best_rf}")
print(f"Train MAE: {train_mae_best_rf}")
print(f"Test MAE: {test_mae_best_rf}")
