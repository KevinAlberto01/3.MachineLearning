#1.IMPORTING LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, LabelEncoder

#2.LOADING DATASET
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Jittering.csv')

#3.ENCODING CATEGORICAL VARIABLES
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

#4.SEPARATING INDEPENDENT(X) AND DEPENDENT VARIABLES(Y)
X = df.drop(columns=['saleprice'])
y = df['saleprice']

#5.SCALING THE FEATURES
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

#6.SPLITTING DATASET INTRO TRAINING AND TEST
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#7.TRAINING LINEAR REGRESSION MODEL
model = LinearRegression()
model.fit(X_train, y_train)

#8.GENERATING PREDICTIONS
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#9.EVALUATING THE MODEL
print(f"Linear Regression - Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred))}")
print(f"Linear Regression - Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred))}")
print(f"Linear Regression - Train R²: {r2_score(y_train, y_train_pred)}")
print(f"Linear Regression - Test R²: {r2_score(y_test, y_test_pred)}")
