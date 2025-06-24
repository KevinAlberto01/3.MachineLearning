#1.IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

#2.LOADING DATASET (JITTERING DATA)
df = pd.read_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Jittering.csv')

#3.FEATURE ENGINEERING
df['TotalBathrooms'] = df['full_bath'] + df['half_bath'] * 0.5
df['HouseAge'] = 2025 - df['year_built']
df['PricePerSF'] = df['saleprice'] / df['gr_liv_area']

#4.CODING OF CATEGORICAL VARIABLES 
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

#5.DEFINING INDENPENDENT(X) AND DEPENDENT VARIABLES(Y)
X = df.drop(columns=['saleprice'])
y = df['saleprice']

#6.SCALING NUMERICAL VARIABLES
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

#7.DIVIDE THE DATASET INTRO TRAINING AND TEST
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#8.DEFINE MODELS 
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

#9.TRAIN AND EVALUATE MODELS
results = []

#9.1 Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        'Model': name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train R²': train_r2,
        'Test R²': test_r2
    })

#10.SAVE RESULTS IN A CSV
results_df = pd.DataFrame(results)
print(results_df)

#10.1 Save results to CSV
results_df.to_csv('/home/kevin/Desktop/Kevin/3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/GenerativeDataAugmentation(Jittering).csv', index=False)

#11.GRAPHICAL MODEL COMPARISON
plt.figure(figsize=(10, 6))
plt.bar(results_df['Model'], results_df['Test RMSE'], color=['skyblue', 'lightgreen', 'coral', 'gold'])
plt.ylabel('Test RMSE')
plt.title('Model Comparison - Test RMSE')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.7)

#11.1 Show the plot
plt.show()
