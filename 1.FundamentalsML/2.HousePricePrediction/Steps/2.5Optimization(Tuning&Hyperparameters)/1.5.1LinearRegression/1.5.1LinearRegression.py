#1.IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#2.LOAD AND PREPARE DATA
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')
X = pd.get_dummies(df.drop(columns=['saleprice']), drop_first=True)
y = df['saleprice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3.DATA STANDARDIZATION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#4.MODEL SELECTION
# For LinearRegression:
model = LinearRegression()
# For Ridge Regression:
# model = Ridge(alpha=1.0)  # You can change alpha value here for regularization

#5.CROSS-VALIDATION
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
mean_cv_score = np.mean(cv_scores)

#6.TRAINING AND EVALUATION
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

#7.EVALUATION MODEL
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"\nCross-Validation Mean RMSE: {-mean_cv_score**0.5:.2f}")  # Showing RMSE from CV
print(f"Linear Regression Test RMSE: {rmse:.2f}")
print(f"Linear Regression Test R²: {r2:.2f}")
