<p align = "center" >
    <h1 align = "Center"> Decision Tree Regression</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

This program implements a **decision tree model** (DecisionTreeRegressor) to predict house prices based on various features. 
The model is trained with a dataset, evaluated using performance metrics, and visualized with a scatter plot that compares the actual and predicted values.
This code allows evaluating the ability of decision trees to model the relationship between a house's features and its price, with the possibility of detecting overfitting if the training performance is much better than the test performance. 

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Real vs Predicted </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/1.RealvsPredicted.png" width="4000"/>

<p align = "center" >
    <h4 align = "Center"> RMSE vs R2 </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/2.TrainTest.png" width="4000"/>


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (PENDING)💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**pandas:** Handling data in tabular format (DataFrame). <br> **train_test_split:** Split the data into training and test sets.<br> **DecisionTreeRegressor:** Regression model based on decision trees.<br> **mean_squared_error, r2_score:** Metrics to evaluate the model's performance. <br> **matplotlib.pyplot:** To plot the results.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D1.ImportLibraries.png" width="4000"/>|
Load the clean dataset of house prices from the specified path.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D2.LoadData.png" width="4000"/>|
**`X`:** It contains all the columns except saleprice (which is the house price). <br> **`y`:** It contains only the saleprice column, which is the variable we want to predict.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D3.SeparateFeatures.png" width="4000"/>|
**`pd.get_dummies()`:** Converts categorical variables into numerical values using "One-Hot Encoding". <br> **`drop_first=True:`** It removes one category from each categorical variable to avoid collinearity.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D4.OneHotEncoding.png" width="4000"/>|
**80%** of the data is used for training (**`X_train`**, **`y_train`**).<br> **20%** is used to evaluate the model (**`X_test`**, **`y_test`**). <br> **`random_state=42`:** Ensures that the split is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D5.SplitDataIntoTraining.png" width="4000"/>|
**DecisionTreeRegressor:** Decision tree-based model for regression. <br> **`fit()`:** Fits the model to the training data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D6.CreateaANDTrain.png" width="4000"/>|
**`predict(X_train)`:** Predicts the saleprice values in the training data.<br> **`predict(X_test)`:** Predicts the saleprice values in the test data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D7.PredictionsTraining.png" width="4000"/>|
**Root Mean Squared Error (RMSE):** Measures how much the predictions deviate from the actual value. <br> **`squared=False`** returns the square root of the mean squared error. <br> **R² Score (Coefficient of determination):**  It measures how well the model explains the variability of **`y`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D8.EvaluateModelsPerformance.png" width="4000"/>|
Show the **RMSE** and **R²** values for the training and test sets.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D9.PrintResults.png" width="4000"/>|
**plt.scatter(y_test, y_test_pred, color="#87CEEB")**: Creates a scatter plot where: <br> **X-axis:** Actual values of saleprice (y_test). <br> **Y-axis:** Predicted values of saleprice (y_test_pred). <br> **color="#87CEEB":** Use a light blue tone (Sky Blue). <br> **alpha=0.6:** Adjusts the transparency of the points. <br> **edgecolors='k':** Adds black borders to the points. <br> **plt.xlabel()** and **plt.ylabel():** Axis labels. <br> **plt.title():** Title of the graph. <br> **plt.show():** Displays the graph.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Models/2.Decision%20Tree%20Regressor/Images/D10.VisualizeResults.png" width="4000"/>|
