#1.IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#2.LOAD DATA
file_path = '/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/AmesHousing_cleaned.csv'
df = pd.read_csv(file_path)

#3.SEPARATE THE FEATURES (X) AND THE TARGET VARIABLE (Y)
X = df.drop(columns=['saleprice'])
y = df['saleprice']

#4.ONE-HOT ENCODING FOR CATEGORICAL COLUMNS
X = pd.get_dummies(X, drop_first=True)

#5.SPLIT THE DATA INTO TRAINING AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#6.CREATE AND TRAIN THE MODEL
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

#7.MAKE PREDICTIONS ON THE TRAINING AND TEST DATA
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#8.EVALUATE THE MODEL'S PERFORMANCE
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

#9.PRINT THE RESULTS
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")

#10.VISUALIZE THE RESULTS
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color="#87CEEB", alpha=0.6, edgecolors='k')
plt.xlabel('Real SalePrice')
plt.ylabel('Predicted SalePrice')
plt.title('Real vs Predicted SalePrice (Decision Tree)')
plt.show()