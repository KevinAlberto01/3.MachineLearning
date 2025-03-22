import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. LOAD DATASET WITH JITTERING
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Jittering.csv')

# 2. SEPARATE INDEPENDENT (x) AND DEPENDENT (Y) VARIABLES
x = df.drop(columns=['saleprice'])
y = df['saleprice']

# 3. SCALE THE DATA
scaler = RobustScaler()
x_scaled = scaler.fit_transform(x)

# 4. SPLIT INTO TRAINING AND TEST SETS
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# 5. DEFINE MODELS
MODELS = {
    "LINEAR REGRESSION": LinearRegression(),
    "DECISION TREE": DecisionTreeRegressor(random_state=42),
    "RANDOM FOREST": RandomForestRegressor(n_estimators=100, random_state=42),
    "K-NEIGHBORS": KNeighborsRegressor(n_neighbors=5)
}

# 6. TRAIN AND EVALUATE EACH MODEL
for name, model in MODELS.items():
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f'📌 {name}:')
    print(f'   ✅ TRAIN RMSE: {train_rmse:.2f} | TEST RMSE: {test_rmse:.2f}')
    print(f'   ✅ TRAIN R²: {train_r2:.2f} | TEST R²: {test_r2:.2f}')
    print('-' * 50)
