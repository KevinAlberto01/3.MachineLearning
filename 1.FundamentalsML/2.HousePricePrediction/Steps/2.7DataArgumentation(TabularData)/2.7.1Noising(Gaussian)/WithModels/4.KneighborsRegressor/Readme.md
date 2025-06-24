<p align = "center" >
    <h1 align = "Center"> 4.Kneighbors Regressor</h1>
</p>

Implements a K-Nearest Neighbors Regressor (KNN) model to predict housing prices using the Ames Housing dataset. Feature engineering techniques, data scaling and data augmentation with Gaussian noise are applied to improve the robustness of the model.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/4.KneighborsRegressor/Images/Data.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas** and **numpy** to handle and process data. <br> **train_test_split** to split the data into training and testing. <br> **RobustScaler** to scale the data and make it more robust to outliers. <br> **KNeighborsRegressor** to train the KNN model. <br> **mean_squared_error** and **r2_score** to evaluate the model performance.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/4.KneighborsRegressor/Images/1.ImportLibraries.png" width="4000"/>|
|The **AmesHousing_cleaned.csv** dataset is loaded with information about houses and their characteristics.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/4.KneighborsRegressor/Images/2.DatasetLoading.png" width="4000"/>|
|Three new features are created: <br> **TotalBathrooms:** Total number of bathrooms considering that **`half_bath`** equals 0.5 of a full bath. <br> **HouseAge:** Age of the house in years, subtracting the construction year from 2025. <br> **PricePerSF:** Price per square foot (**`saleprice`** divided by **`gr_liv_area`**).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/4.KneighborsRegressor/Images/3.FeatureEngineering.png" width="4000"/>|
|The **independent variables (X)** are defined by eliminating **`saleprice`** (target). <br> Categorical variables are converted into numerical variables with **`pd.get_dummies()`**, eliminating the first category (drop_first=True) to avoid collinearity. <br> X is **scaled** with **`RobustScaler`**, which reduces the effect of outliers.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/4.KneighborsRegressor/Images/4.DataPreparation.png" width="4000"/>|
|A function **`add_gaussian_noise()`** is defined that adds Gaussian noise to each numerical column. <br> Multiply the standard deviation of each column by **`noise_factor=0.01`** (1% of the standard deviation).<br> Use this function to obtain **`X_augmented`**.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/4.KneighborsRegressor/Images/5.DataAugmentation.png" width="4000"/>|
|The dataset is divided into **80% training** and **20% test** (test_size=0.2). br> **`Random_state=42`** is used to guarantee reproducibility.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/4.KneighborsRegressor/Images/6.DivisionIntoTrainingTest.png" width="4000"/>|
|An instance of the **K-Nearest Neighbors Regressor (KNN)** model is created. <br> It is trained with **`X_train`** and **`y_train`**.
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/4.KneighborsRegressor/Images/7.KNNModelTraining.png" width="4000"/>|
|Predictions are obtained for the training and test sets. <br> Metrics are calculated: <br> **RMSE (Root Mean Squared Error):** Measures the average error of the predictions. <br> **R² Score:** Indicates how well the model explains the variability of the data (1 is perfect, negative values indicate a bad fit). <br> The model results are printed.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/4.KneighborsRegressor/Images/8.ModelPredictionEvaluation.png" width="4000"/>|