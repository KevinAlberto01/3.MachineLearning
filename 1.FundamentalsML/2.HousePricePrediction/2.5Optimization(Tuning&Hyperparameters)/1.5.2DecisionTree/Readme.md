<p align = "center" >
    <h1 align = "Center"> 1.5.2 Decision Tree </h1>
</p>

<p align = "center" >
    <h2 align = "Center">📜 Description 📜</h2>
</p>

Implements a decision tree model to predict house prices using the “AmesHousing_cleaned.csv” dataset.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.2DecisionTree/Images/DecisionResults.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** To handle the data in the form of DataFrame. <br> **train_test_split:** To split the data into training and test. <br> **GridSearchCV:** To find the best hyperparameters of the model. <br> **DecisionTreeRegressor:** Regression model based on decision trees. <br> **mean_squared_error**, **r2_score:** Model evaluation metrics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.2DecisionTree/Images/D1.ImportLibraries.png" width="4000"/>|
|The data set is loaded from a CSV file located in the specified path.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.2DecisionTree/Images/D2.DataLoading.png" width="4000"/>|
|The column **`saleprice`** is removed from the DataFrame, since it is the target variable. <br> Categorical variables are converted to dummy variables with **`pd.get_dummies()`**. <br> **`drop_first=True`** removes one of the categories to avoid multicollinearity. **`y`** stores the target variable (**`saleprice`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.2DecisionTree/Images/D3.DataPreprocessing.png" width="4000"/>|
|The data set is divided into training (**80%**) and test (**20%**). <br> **`random_state=42`** ensures that the split is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.2DecisionTree/Images/D4.DataSetSplitting.png" width="4000"/>|
|A dictionary is defined with the hyperparameters to be optimized: <br> **`max_depth`:** Maximum depth of the decision tree. <br> **`min_samples_split`:** Minimum number of samples required to split a node.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.2DecisionTree/Images/D5.HyperparameterGrid.png" width="4000"/>|
|**`GridSearchCV`** is used to test all combinations of hyperparameters. <br> **`cv=5`:** Cross validation with 5 partitions is used. <br> **`scoring='neg_mean_squared_error'`:** Negative **mean squared error** is used as metric. <br> **`fit(X_train, y_train)`:** The model is trained with the training data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.2DecisionTree/Images/D6.SearchingBestHyper.png" width="4000"/>|
The model with the best hyperparameters found by **`GridSearchCV`** is obtained.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.2DecisionTree/Images/D7.SelectionBestModel.png" width="4000"/>|
|The **best hyperparameters** found are printed. <br> Predictions are made with **`best_model.predict(X_test)`**. <br> Two evaluation metrics are calculated: <br> **RMSE (Root Mean Squared Error):** Average error of the predictions. <br> **R² (Coefficient of Determination):** Indicates how well the model explains the variability of the data (1 is the best value).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.2DecisionTree/Images/D8.ModelPredictionEvaluation.png" width="4000"/>|