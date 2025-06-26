<p align = "center" >
    <h1 align = "Center"> 1.5.4 KNeighbors Regressor </h1>
</p>

<p align = "center" >
    <h2 align = "Center">📜 Description 📜</h2>
</p>
Implements a regression model based on K-Nearest Neighbors (KNN) to predict house prices. A dataset is loaded, preprocessed by converting categorical variables to numerical variables and split into training and test data. Then, GridSearchCV is used to optimize the hyperparameters of the model by testing different values of number of neighbors and weighting methods. Finally, the model is evaluated using metrics such as RMSE (root mean square error) and R² to measure its accuracy.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.4KNeighborsRegressor/Images/KNeighborsResults.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (PENDING)💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** For data manipulation. <br> **train_test_split:** To split the data into training and test.<br> **GridSearchCV:** To optimize the KNeighborsRegressor hyperparameters. <br> **KNeighborsRegressor:** Regression model based on nearest neighbors. <br> **mean_squared_error**, **r2_score:** To evaluate the model performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/1.DeclareteLibraries.png" width="4000"/>|
|Load the dataset from a CSV file.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/K2.DataLoading.png" width="4000"/>|
|**`df.drop(columns=['SalePrice'])`:** Removes the **`SalePrice`** column, because this is the target variable (**`y`**). <br> **`pd.get_dummies(..., drop_first=True)`**: Converts categorical variables to one-hot encoding numeric variables and drops one category to avoid collinearity. <br> **`y = df['SalePrice']`**: The target variable is assigned.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/K3.DataPreparation.png" width="4000"/>|
|Splits the data into **80%** for training and **20%** for testing. <br> **`random_state=42`** ensures that the division is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/K4.DivisionTraSet.png" width="4000"/>|
|Defines the hyperparameters to be optimized by **GridSearchCV**: <br> **`n_neighbors`:** Number of neighbors to consider (3, 5, 7 or 9). <br> **`weights`:** Type of weighting of the neighbors: <br> **` 'uniform'`**: All neighbors have the same weight. <br> **`'distance'`:** Closest neighbors have more weight.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/K5.DefinitionHyper.png" width="4000"/>|
|**`GridSearchCV(...)`** creates a hyperparameter search model with: <br> **`'cv=5'`**: 5-fold cross-validation. <br> **`scoring='neg_mean_squared_error'`:** Negative mean squared error (MSE) is optimized. <br> **`.fit(X_train, y_train)`:** Find the best combination of hyperparameters by training and evaluating on **`X_train`** and **`y_train`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/K6.SearchingHyper.png" width="4000"/>|
|**`grid_search.best_estimator_`**: Gets the best model found. <br> **`grid_search.best_params_`:** Prints the optimal hyperparameters.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/K7.ObtainingBestModel.png" width="4000"/>|
|Predictions are generated on the test set (**`X_test`**). <br> **`mean_squared_error(y_test, y_pred, squared=False)`:** Calculates the root mean squared error (RMSE), which measures the average difference between the predictions and the actual values. <br> **`r2_score(y_test, y_pred)`:** Calculates the coefficient of determination. 𝑅2, which indicates how well the model explains the variability of the data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/K8.ModelEvaluation.png" width="4000"/>|