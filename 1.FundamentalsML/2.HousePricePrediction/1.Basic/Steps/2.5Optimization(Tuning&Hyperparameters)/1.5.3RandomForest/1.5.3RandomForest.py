#1.IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#2.LOAD DATA
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

#3.DATA PREPROCESSING
X = pd.get_dummies(df.drop(columns=['SalePrice']), drop_first=True)
y = df['SalePrice']

#4.SPLITTING DATA INTO TRAINING AND TESTING 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#5.DEFINING THE HYPERPARAMETER GRID 
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30, None]
}

#6.PERFORMING HYPERPARAMETER SEARCH WITH GRIDSEARCHCV
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

#7.OBTAIN THE BEST MODELS
best_model = grid_search.best_estimator_

#8.DISPLAY THE BEST HYPERPARAMETERS
print(f"\nBest Params: {grid_search.best_params_}")

#9.MAKE PREDICTIONS WITH THE BEST MODEL 
y_pred = best_model.predict(X_test)

#10.EVALUATE THE MODEL 
print(f"Random Forest Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"Random Forest Test R²: {r2_score(y_test, y_pred):.2f}")
