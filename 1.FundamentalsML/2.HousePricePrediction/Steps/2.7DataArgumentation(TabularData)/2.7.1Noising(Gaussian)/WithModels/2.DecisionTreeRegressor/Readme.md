<p align = "center" >
    <h1 align = "Center"> 2.Decision Tree Regressor</h1>
</p>

Implements a DecisionTreeRegressor model to predict the price of houses, applying data preprocessing techniques such as feature engineering, scaling with RobustScaler and data augmentation with jittering (Gaussian noise).

The evaluation of the model allows comparing its performance on training and test data, helping to detect overfitting if the model has a very high R² in training but low in testing.


<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/Data.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas** and **numpy** for data manipulation. <br> **train_test_split** for splitting data into training and testing. <br> **RobustScaler** to normalize features and reduce the impact of outliers. <br> **DecisionTreeRegressor** as machine learning model. <br> **mean_squared_error** and **r2_score** to evaluate model performance.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/1.ImportLibraries.png" width="4000"/>|
|Here you load the dataset containing information about different houses, with characteristics such as year of construction, number of bathrooms, living area, etc.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/2.DataLoading.png" width="4000"/>|
|New useful features are created for the model: <br> **TotalBathrooms:** Sums the full bathrooms and half of the half bathrooms. <br> **HouseAge:** Calculated by subtracting the year of construction from 2025 to obtain the age of the house. <br> **PricePerSF:** Calculated by dividing the sales price by the living area.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/3.FeatureEngineering.png" width="4000"/>|
|The predictor variables **`X`** and the target variable **`y`** (house price) are separated. <br> The categorical variables are converted into numerical variables with **`pd.get_dummies()`**. <br> **RobustScaler** is applied to scale the data and reduce the influence of outliers.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/4.DataPreparation.png" width="4000"/>|
|Here we implement **jittering**, adding Gaussian noise to each column of **`X_scaled`** with a standard deviation of 1% (**`noise_factor=0.01`**). This helps to improve the generalization of the model by making it more robust to small variations in the data.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/5.DataAugmentation.png" width="4000"/>|
|The dataset is divided into **80% for training and 20% for testing**, ensuring reproducibility with random_state=42.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/6.DivisionDataSet.png" width="4000"/>|
|A decision **tree for regression** is trained. This model is able to capture nonlinear relationships in the data by partitioning the feature space into regions based on decision rules.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/7.DecisionTreeRegressor.png" width="4000"/>|
|Predictions are generated for both the training and test sets.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/8.ModelPredictions.png" width="4000"/>|
|The following metrics are calculated: <br> **RMSE (Root Mean Squared Error):** measures the root mean squared error on the same scale as the house price. <br> **R² (Coefficient of Determination):** Measures how well the model explains the variability of the data (1 is perfect, 0 means it explains nothing).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/9.ModelEvaluation.png" width="4000"/>|
|There is an **error in the printing** here, as the model is a decision tree and not linear regression. The message should read: <br> This error does not affect the performance of the code, but may cause confusion when interpreting the results.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/2.DecisionTreeRegressor/Images/10.PrintoutResults.png" width="4000"/>|