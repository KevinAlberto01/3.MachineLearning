import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

#1.Load and Prepare Data
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv')
X = pd.get_dummies(df.drop(columns=['SalePrice']), drop_first=True)
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2.Standardize (Optional, some prefer to do it for LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#3.Model
model = LinearRegression()

#4.Training and Evaluation
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print(f"\nLinear Regression Test RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"Linear Regression Test R²: {r2_score(y_test, y_pred):.2f}")
