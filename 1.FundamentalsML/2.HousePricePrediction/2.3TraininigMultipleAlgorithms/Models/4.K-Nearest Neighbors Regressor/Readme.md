<p align = "center" >
    <h1 align = "Center"> K-Nearest Neighbors Regressor</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

The code implements a K-Nearest Neighbors (KNN) Regressor model to predict house prices based on different features.  It evaluates the model's performance and visualizes the relationship between the actual and predicted values.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Real vs Predicted </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/1.RealvsPredict.png" width="4000"/>

<p align = "center" >
    <h4 align = "Center"> RMSE vs R2 </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/2.TrainTest.png" width="4000"/>


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**pandas:** Data handling and manipulation. <br> **train_test_split:** Splitting the dataset into training and test data. <br> **KNeighborsRegressor:** Implementation of the KNN algorithm for regression. <br> **mean_squared_error, r2_score:** Metrics to evaluate the model. <br> matplotlib.pyplot: Graph generation.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K1.ImportLibraries.png" width="4000"/>|
The CSV file containing house price data and their characteristics is read.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K2.LoadData.png" width="4000"/>|
**`X`** contains the features of the house, excluding the price column. <br> **`y`** it is the target variable, which represents the house price.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K3.SeparationFeature.png" width="4000"/>|
Convert categorical variables into numerical variables using **One-Hot Encoding**, removing the first category to avoid multicollinearity.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K4.OneHotEncoding.png" width="4000"/>|
The dataset is divided into: <br> **80%** for training (**`X_train`**, **`y_train`**). <br> **20%** for testing (**`X_test`**, **`y_test`**). <br> The parameter **`random_state=42`** ensures that the split is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K5.DataSetSplitting.png" width="4000"/>|
A **K-Nearest Neighbors (KNN)** model is instantiated without specifying n_neighbors, so it will use the default value (5 neighbors). <br> The model is trained with the training data (**`X_train`**, **`y_train`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K6.ModelCreationTraining.png" width="4000"/>|
Predictions are made on the training data (**`y_train_pred`**). <br> Predictions are made on the test data (**`y_test_pred`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K7.PredictionModel.png" width="4000"/>|
The metrics are calculated to evaluate performance: <br> **RMSE (Root Mean Squared Error):** measures the average error of the predictions in units of house price. <br> **R² (Coefficient of Determination):** measures how well the model explains the variability of the data (1 is the best value, 0 is random).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K8.ModelEvaluation.png" width="4000"/>|
The **RMSE** and **R²** values are printed for both **training** and **testing**. <br> If the **test RMSE** is **much higher** than the training RMSE, the model could be **overfitting.**| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K9.PrintingResults.png" width="4000"/>|
A **scatter plot** is generated comparing **actual prices** vs. **predicted prices**. <br> **Color** **`#87CEEB`:** Light blue is used for the dots. <br> **Points with black edges** (**`edgecolors='k'`**). <br> **If the points are aligned in a perfect diagonal**, it means that the model predicts with great accuracy.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/4.K-Nearest%20Neighbors%20Regressor/Images/K10.VisualizationResults.png" width="4000"/>|