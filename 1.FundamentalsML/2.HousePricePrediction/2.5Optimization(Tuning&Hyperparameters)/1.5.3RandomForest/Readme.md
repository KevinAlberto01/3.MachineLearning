<p align = "center" >
    <h1 align = "Center"> 1.5.3 Random Forest </h1>
</p>

<p align = "center" >
    <h2 align = "Center">📜 Description 📜</h2>
</p>

Trains a Random Forest Regressor model to predict house prices using a preprocessed dataset

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/RandomResults.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** To load and manipulate the data. <br> **train_test_split:** To split the data into training and test. <br> **GridSearchCV:** To perform **grid search** and find the best hyperparameters of the model. <br> **RandomForestRegressor:** The random forest model for regression. <br> **mean_squared_error**, **r2_score:** Metrics to evaluate model performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/R1.ImportLibraries.png" width="4000"/>|
|Upload the CSV file containing the house data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/R2.LoadData.png" width="4000"/>|
|**`df.drop(columns=['SalePrice'])`:** The house price values are dropped from the input data **`X`**, since it is the variable we want to predict. <br> **`pd.get_dummies(..., drop_first=True)`:** Converts categorical variables to numeric variables by **one-hot encoding**, dropping one category to avoid multicollinearity. <br> **`y = df['SalePrice']`:** Defines the target variable **`y`** as the price of the house. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/R3.DataPreprocessing.png" width="4000"/>|
|**`test_size=0.2`:** Uses 20% of the data for testing and 80% for training. <br> **`random_state=42`:** Ensures that the division is always the same when running the code.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/R4.SplittingData.png" width="4000"/>|
|**`n_estimators`:** Number of trees in the forest. <br> **`max_depth`:** Maximum depth of the trees (how many splits they can make before stopping). **`None`** allows unlimited growth.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/R5.DefiningHyperparameter.png" width="4000"/>|
|**`GridSearchCV(...)`:** Performs cross-validation with 5 partitions (**`cv=5`**). <br> **`scoring='neg_mean_squared_error'`:** Maximizes accuracy by minimizing the mean square error. <br> **`grid_search.fit(X_train, y_train)`:** Train the model with different combinations of hyperparameters.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/R6.PerformingHyperparameter.png" width="4000"/>|
|Extract the model with the **best hyperparameters** found in the search.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/R7.ObtainBestModels.png" width="4000"/>|
|Prints the optimal values of **`n_estimators`** and **`max_depth`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/R8.DisplayBestHyper.png" width="4000"/>|
|Uses the optimized model to predict house prices on test data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest/Images/R9.MakePredictions.png" width="4000"/>|
|**RMSE (Root Mean Squared Error):** Indicates the average error in the predictions. <br> **R² (Coefficient of Determination):** Measures how well the model explains the variability of the data (values close to 1 indicate a good model).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.3RandomForest//Images/R10.EvaluateModel.png" width="4000"/>|