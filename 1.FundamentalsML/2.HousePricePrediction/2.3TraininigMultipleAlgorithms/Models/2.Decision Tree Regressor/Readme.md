<p align = "center" >
    <h1 align = "Center"> Linear Regression</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

This program trains a **Linear Regression model** to predict the sale price of houses (**SalePrice**) based on various features.  Data processing techniques, splitting into training and test sets, and model evaluation using metrics such as **RMSE** and **R²** are employed.  Additionally, a scatter plot is generated to visualize the relationship between the actual and predicted values.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Real vs Predicted </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/1.RealvsPredicted.png" width="4000"/>

<p align = "center" >
    <h4 align = "Center"> RMSE vs R2 </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/2.TrainTest.png" width="4000"/>


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (PENDING)💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
The necessary libraries are imported: **`Pandas`** for handling data, **`sklearn`** for regression and evaluation, and **`matplotlib.pyplot`** for plotting. <br> The dataset is loaded from the specified path.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L1.ImportLibraries.png" width="4000"/>|
**`X`:** Contains all the columns except **`SalePrice`** (the model features). <br> **`y`:** Contains only **`SalePrice`** (the target variable to predict).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L2.LoadData.png" width="4000"/>|
**One-Hot Encoding** is used to convert categorical variables into numerical ones. <br> **`drop_first=True`** removes a redundant category to avoid collinearity.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L3.SeparateFeatureTarget.png" width="4000"/>|
The dataset is divided into **80% for training** and **20% for testing**. <br> **`random_state=42`** ensures that the results are reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L4.OneHotEncoding.png" width="4000"/>|
A **Linear Regression** model is initialized with **`LinearRegression()`**. <br> **`model.fit(X_train, y_train)`** trains the model using the training data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L5.TrainTest.png" width="4000"/>|
**Predictions** are made on the training and test data. <br> The evaluation **metrics are calculated:** <br> **RMSE (Root Mean Squared Error):** Root mean squared error, measures how far the predictions are from the actual values. <br> **R² Score:** Indicates what percentage of the variability of **`SalePrice`** is explained by the characteristics (**`X`**).  A value close to 1 indicates a good fit.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L6.Model.png" width="4000"/>|
The metrics are printed to evaluate the model's performance on the training and test data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L7.Prediction.png" width="4000"/>|
The following evaluation metrics are calculated: <br> **RMSE (Root Mean Squared Error)**: Average prediction error. <br> **R² (Coefficient of Determination)**: Indicates how well the model explains the variability of the data (0 to 1, the higher the better).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L8.EvaluateModel.png" width="4000"/>|
The metrics are printed to analyze the model's performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L9.ShowResults.png" width="4000"/>|
**Scatter plot** where: <br>**X-axis:** Actual house prices (**`SalePrice`**). <br> **Y-Axis:** Price predicted by the model. <br> **Color #87CEEB (Sky Blue)** for the points. <br> **Objective:** If the model were perfect, all the points would fall on the **`line y = x`.**|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/1.Linear%20Regression/Images/L10Graph.png" width="4000"/>|