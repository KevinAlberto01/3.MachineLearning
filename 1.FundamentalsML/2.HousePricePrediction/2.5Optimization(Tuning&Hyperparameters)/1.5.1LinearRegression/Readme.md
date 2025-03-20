<p align = "center" >
    <h1 align = "Center"> 1.5.1 Linear Regression </h1>
</p>

<p align = "center" >
    <h2 align = "Center">📜 Description 📜</h2>
</p>
Linear regression doesn't have as many hyperparameters to tune, but you can check how data standardization affects it.  

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.1LinearRegression/Images/LinearResults.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** For data manipulation. <br> **train_test_split:** To split the data into training and test. <br> **cross_val_score:** To evaluate the model with cross validation. <br> **LinearRegression and Ridge:** Linear regression models (with and without regularization). <br> **StandardScaler:** To standardize the data. <br> **mean_squared_error**, **r2_score:** To evaluate model performance. <br> **numpy:** For numerical calculations.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.1LinearRegression/Images/L1.ImportLibraries.png" width="4000"/>|
|The CSV file is read. <br> Unused columns are removed (**`saleprice`** is the target variable **y**). <br> Convert categorical variables to numerical variables with **`pd.get_dummies()`**. <br> The dataset is divided into **80% training** and **20% testing**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.1LinearRegression/Images/L2.LoadPrepareData.png" width="4000"/>|
|**StandardScaler** is applied to normalize the features. <br> The scaler is adjusted on the training data and the transformation is applied on the test data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.1LinearRegression/Images/L3.DataStandardization.png" width="4000"/>|
**Linear regression**(**`LinearRegression()`**): Regression model without penalty. <br> **Ridge Regression** (**`Ridge(alpha=1.0)`**) (optional): Used when there is collinearity in the data to avoid overfitting.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.1LinearRegression/Images/L4.ModelSelection.png" width="4000"/>|
|**Cross validation** with **`cv=5`** is applied to evaluate the model on different subsets of the training set. <br> **`Neg_mean_squared_error`** is used (negative is used because scikit-learn maximizes the metric). <br> The error is averaged to obtain a representative value.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.1LinearRegression/Images/L5.CrossValidation.png" width="4000"/>|
The model is trained with **`X_train_scaled`** and **`y_train`**. <br> Predictions are generated on the test data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.1LinearRegression/Images/L6.TrainingEvaluation.png" width="4000"/>|
|**RMSE (Root Mean Squared Error):** Measures the average error of the model. <br> **R² (Coefficient of Determination):** Indicates how well the model explains the variability of the data. <br> RMSE results are printed in cross-validation and in the test set.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/1.5.1LinearRegression/Images/L7.EvaluationModel.png" width="4000"/>|
