import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')
X = pd.get_dummies(df.drop(columns=['saleprice']), drop_first=True)
y = df['saleprice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print(f"\nBest Params: {grid_search.best_params_}")
y_pred = best_model.predict(X_test)

print(f"Decision Tree Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"Decision Tree Test R²: {r2_score(y_test, y_pred):.2f}")
