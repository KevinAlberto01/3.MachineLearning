<p align = "center" >
    <h1 align = "Center"> Random Forest Regressor</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

This program implements a **Random Forest Regressor** model to predict house prices using structured data. A clean dataset is loaded, categorical variables are transformed using **One-Hot Encoding**, and it is split into training and test sets. Subsequently, a **Random Forest** model is trained, predictions are made, and the model's performance is evaluated using metrics such as **RMSE (Root Mean Squared Error)** and **R² (Coefficient of Determination)**. Finally, a scatter plot is generated to visualize the relationship between the actual values and the values predicted by the model.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Real vs Predicted </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/1.RealvsPredict.png" width="4000"/>

<p align = "center" >
    <h4 align = "Center"> RMSE vs R2 </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/2.TrainTest.png" width="4000"/>


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (PENDING)💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**pandas:** To handle the dataset. <br> **train_test_split:** To split the data into training and test sets. <br> **RandomForestRegressor:** Random Forest model for regression. <br> **mean_squared_error, r2_score:** To evaluate the model's performance. <br> **matplotlib.pyplot:** To visualize the results with a graph.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R1.ImportLibraries.png" width="4000"/>|
The dataset is loaded from a CSV file.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R2.LoadTheData.png" width="4000"/>|
**`X`:** It contains all the columns except **`'saleprice'`**, as these will be the input features. <br> **`y`:** It contains only the target variable **`'saleprice'`**, which represents the price of the houses.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R3.SeparationFeaturesTargetVariables.png" width="4000"/>|
Since some variables can be categorical (for example, type of house, area, construction materials), **One-Hot Encoding** is used to convert them into numerical values. <br> **`drop_first=True`:** It avoids multicollinearity by eliminating one of the categories for each categorical variable.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R4.TransformationCategoricalVariables.png" width="4000"/>|
Here, the data is divided into two parts: <br> **80% for training**(**`X_train`**, **`y_train`**). <br> **20% for testing** (**`X_test`**, /**`y_test`**). <br> **`random_state=42`** is used to ensure that the split is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R5.SplittingTraining.png" width="4000"/>|
A **RandomForestRegressor** is created, which is a set of decision trees trained on random subsets of the data. <br> The model is trained with model.fit(**`X_train`**, **`y_train`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R6.CreationTraining.png" width="4000"/>|
Here the model predicts the prices of houses in: <br> The training set (**`y_train_pred`**). <br> The test set (**`y_test_pred`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R7.GenerationPredictions.png" width="4000"/>|
Key metrics are calculated: <br> **RMSE (Root Mean Squared Error):** <br> Measures the average error between the actual and predicted values. <br> The smaller, the better. <br> **R² (coefficient of determination):** <br> Measures how well the model explains the variability of the data. <br>**1.0** means perfect prediction. <br>**Values close to 0** indicate a poor model.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R8.ModelEvaluation.png" width="4000"/>|
Here the values obtained in the evaluation are printed.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R9.PrintResults.png" width="4000"/>|
A **scatter plot** is created to compare the actual values (**`y_test`**) vs. the predicted values (**`y_test_pred`**). <br> **`color="#87CEEB"`:** Use a light blue tone in the dots. <br> **alpha=0.6:** Makes the points more transparent. <br> *edgecolors='k':** Adds black borders to the points.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/3.Random%20Forest%20Regressor/Images/R10.VisualizeResults.png" width="4000"/>|
