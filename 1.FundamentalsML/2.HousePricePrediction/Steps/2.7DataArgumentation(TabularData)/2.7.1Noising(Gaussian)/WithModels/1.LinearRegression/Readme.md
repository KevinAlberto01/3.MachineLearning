<p align = "center" >
    <h1 align = "Center"> 1.Lineal Regression</h1>
</p>

This program implements a Linear Regression model to predict housing prices using the AmesHousing dataset. Data preprocessing including feature engineering, normalization and data augmentation with jittering is performed to improve the robustness of the model.


<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/Data.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Noising Gaussian) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** Data handling in DataFrame format. <br> **numpy:** Mathematical operations and Gaussian noise generation. <br> **train_test_split:** Splits data into training and testing. <br> **RobustScaler:** Normalizes data reducing the impact of outliers. <br> **LinearRegression:** Linear regression model. <br> **mean_squared_error, r2_score:** Model evaluation metrics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/1.ImportingLibraries.png" width="4000"/>|
|Loads data from a CSV file containing housing information.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/2.DataSetLoading.png" width="4000"/>|
|Three new features are created to improve the model: <br> **TotalBathrooms:** Considers full and half bathrooms. <br> **HouseAge:** Calculates the age of the house by subtracting the year of construction from 2025. <br> **PricePerSF:** Price per square foot (useful for normalizing prices).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/3.FeatureEngineering.png" width="4000"/>|
|**X (predictor variables):** **`saleprice`** was eliminated because it is the target variable. <br> **`pd.get_dummies()`:** Converts categorical variables to numeric variables.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/4.DataPreparation.png" width="4000"/>|
|**`RobustScaler`:** Scales data to reduce the impact of outliers. <br> **Transforms values** to a more uniform scale.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/5.NormalizationRobustcaler.png" width="4000"/>|
|**Adds random (Gaussian) noise to each numerical column** to improve model generalization.<br> **`noise_factor=0.01`:** Controls the amount of noise added (1% of the standard deviation).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/6.Jittering(Gaussian).png" width="4000"/>|
|**80% for training, 20% for testing.** <br> **`random_state=42`:** Ensures that the results are reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/7.TrainingTestPartition.png" width="4000"/>|
|**Create and train** a linear regression model with the training data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/8.LinearRegressionModel.png" width="4000"/>|
|It generates predictions for both the **training set** and the **test set**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/9.Predictions.png" width="4000"/>|
|**RMSE (Root Mean Squared Error):** Measures how much error the model has on average. <br> **R² (Coefficient of Determination):** Measures how well the model explains the variability of the data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/1.LinearRegression/Images/10.ModelEvaluation.png" width="4000"/>|
