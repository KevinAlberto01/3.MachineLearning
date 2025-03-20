import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')

# Detect and encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
print(f"Linear Regression - Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred))}")
print(f"Linear Regression - Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred))}")
print(f"Linear Regression - Train R²: {r2_score(y_train, y_train_pred)}")
print(f"Linear Regression - Test R²: {r2_score(y_test, y_test_pred)}")
