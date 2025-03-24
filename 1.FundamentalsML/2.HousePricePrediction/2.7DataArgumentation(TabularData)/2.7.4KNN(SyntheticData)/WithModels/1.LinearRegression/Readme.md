<p align = "center" >
    <h1 align = "Center"> 1.Lineal Regression</h1>
</p>

This program implements a Linear Regression model to predict housing prices using the AmesHousing dataset. Data preprocessing including feature engineering, normalization and data augmentation with jittering is performed to improve the robustness of the model.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/Data.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Noising Gaussian) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|Necessary libraries for data loading, preprocessing, modeling, and evaluation are imported. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/1.ImportLibraries.png" width="4000"/>|
|The CSV file is loaded into a DataFrame **`df`** using **pandas.**| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/2.LoadingDataset.png" width="4000"/>|
|Categorical columns in the dataset are identified. <br> **`LabelEncoder`** is used to convert them to numerical values, assigning a number to each category.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/3.EncodingCategoricalVariables.png" width="4000"/>|
|**`X`** contains all columns except **`'SalePrice'`** (input features). <br> **`y`** contains only **`'SalePrice'`**, which is the variable to be predicted.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/4.SeparaVariables.png" width="4000"/>|
|**`RobustScaler()`** is used to normalize the data and make it more resistant to outliers.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/5.ScalingFeatures.png" width="4000"/>|
|The dataset is split into **80% training** and **20% testing sets**. **`random_state=42`** ensures the split is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/6.SplittingDataset.png" width="4000"/>|
|A **`Linear Regression`** model is initialized and trained.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/7.TrainingLinearRegression.png" width="4000"/>|
|Predictions are generated for both training and testing sets.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/8.GeneratingPredictions.png" width="4000"/>|
|**RMSE (Root Mean Squared Error):** Measures the prediction error. <br> **R² (Coefficient of Determination):** Measures how well the model fits the data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/1.LinearRegression/Images/9.EvaluatingModel.png" width="4000"/>|