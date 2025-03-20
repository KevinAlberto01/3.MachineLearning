<p align = "center" >
    <h1 align = "Center"> Training Multiple Algorithms</h1>
</p>


<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>


<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

|Linear Regression |Decision Tree Regressor | Random Forest Regressor| K-Nearest Neighbors Regressor|
|-----------------------------------|------------------------|---------------------|--------------|
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/5.LinealRegression.png" width="4000"/> |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/6.DEcisionTreeRegressor.png" width="4000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/7.RandomForestRegressor.png" width="4000"/> |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/8.KNeighborsRegressor.png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** For data manipulation. <br> **numpy:** For numerical calculations. <br> **train_test_split:** For splitting data into training and testing. <br> **RobustScaler:** For scaling data and reducing the impact of outliers. <br> **LinearRegression**, **DecisionTreeRegressor**, **RandomForestRegressor**, **KNeighborsRegressor**: Regression models to evaluate. <br> **mean_squared_error**, **r2_score**, **mean_absolute_error:** Model evaluation metrics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F1.ImportLibraries.png" width="4000"/>|
|The CSV file is loaded into a DataFrame **`df`**. <br> The available columns in the dataset are printed.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F2.LoadDataset.png" width="4000"/>|
|**Three new features** are created: <br> **TotalBathrooms:** Sum of full bathrooms and half bathrooms (half bath is considered as 0.5). <br> **houseage:** Age of the house in the year 2025. <br> **PricePerSF:** Price per square foot (**`saleprice / gr_liv_area`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F3.ManualFeature.png" width="4000"/>|
|**`saleprice`** is the target variable **`y`** (what we want to predict). <br> Unnecessary columns are eliminated and categorical variables are converted to numeric with **`pd.get_dummies()`**. <br> **RobustScaler** is applied, which is useful to **reduce the impact of outliers**. <br> The data is divided into **80%** training and **20%** testing.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F4.DataPreprocessing.png" width="4000"/>|
**Four models** are defined: <br> **`Linear Regression`:** Basic model without regularization. <br> **`DecisionTreeRegressor`:** Non-linear model that can capture more complex relationships. <br> **`Random Forest (RandomForestRegressor)`:** Set of multiple decision trees to improve accuracy. <br> **`K-Nearest Neighbors (KNN)`:** Model based on nearest neighbors.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F5.DefineModels.png" width="4000"/>|
|Each model is trained with the training data. <br> Predictions are generated for both training and testing. <br> The following evaluation **metrics are calculated:** <br> **RMSE (Root Mean Squared Error):** Root mean squared error. <br> **R² (Coefficient of Determination):** Indicates how well the model explains the variability of the data. <br> **MAE (Mean Absolute Error):** Mean Absolute Error. <br> **MSE (Mean Squared Error):** Mean Squared Error. <br> The results are stored in a **`results list`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F6.ModelTrainingEvaluation.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F7.SavingDisplayingResults.png" width="4000"/>|
