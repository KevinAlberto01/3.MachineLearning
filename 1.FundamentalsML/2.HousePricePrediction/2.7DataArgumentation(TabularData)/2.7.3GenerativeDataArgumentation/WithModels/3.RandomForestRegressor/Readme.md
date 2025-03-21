<p align = "center" >
    <h1 align = "Center"> 3.Random Forest Regressor</h1>
</p>

Implements a K-Nearest Neighbors Regressor (KNN) model to predict housing prices using the Ames Housing dataset. Feature engineering techniques, data scaling and data augmentation with Gaussian noise are applied to improve the robustness of the model.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|Necessary libraries for data loading, preprocessing, modeling, and evaluation are imported. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/1.ImportLibraries.png" width="4000"/>|
|The CSV file is loaded into a DataFrame **`df`** using **pandas.**| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/2.LoadingDataset.png" width="4000"/>|
|Categorical columns in the dataset are identified. <br> **`LabelEncoder`** is used to convert them to numerical values, assigning a number to each category.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/3.EncodingCategoricalVariables.png" width="4000"/>|
|**`X`** contains all columns except **`'SalePrice'`** (input features). <br> **`y`** contains only **`'SalePrice'`**, which is the variable to be predicted.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/4.SeparaVariables.png" width="4000"/>|
|**`RobustScaler()`** is used to normalize the data and make it more resistant to outliers.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/5.ScalingFeatures.png" width="4000"/>|
|The dataset is split into **80% training** and **20% testing sets**. **`random_state=42`** ensures the split is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/6.SplittingDataset.png" width="4000"/>|
|Create a **Random Forest model for regression**. <br> A **Random Forest** is an ensemble of multiple decision trees, each trained on a different sample of the dataset (a "Bagging" method). <br> The predictions of all the **trees are averaged** to obtain the final result, which makes the model more robust and accurate.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/7.TrainingLinearRegression.png" width="4000"/>|
|Predictions are generated for both training and testing sets.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/8.GeneratingPredictions.png" width="4000"/>|
|**RMSE (Root Mean Squared Error):** Measures the prediction error. <br> **R² (Coefficient of Determination):** Measures how well the model fits the data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/3.RandomForestRegressor/Images/9.EvaluatingModel.png" width="4000"/>|