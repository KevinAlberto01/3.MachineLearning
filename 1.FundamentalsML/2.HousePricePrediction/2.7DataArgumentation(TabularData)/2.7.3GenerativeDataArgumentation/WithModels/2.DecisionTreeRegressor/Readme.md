<p align = "center" >
    <h1 align = "Center"> 2.Decision Tree Regressor</h1>
</p>

Implements a DecisionTreeRegressor model to predict the price of houses, applying data preprocessing techniques such as feature engineering, scaling with RobustScaler and data augmentation with jittering (Gaussian noise).

The evaluation of the model allows comparing its performance on training and test data, helping to detect overfitting if the model has a very high R² in training but low in testing.


<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/Data.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|Necessary libraries for data loading, preprocessing, modeling, and evaluation are imported. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/1.ImportLibraries.png" width="4000"/>|
|The CSV file is loaded into a DataFrame **`df`** using **pandas.**| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/2.LoadingDataset.png" width="4000"/>|
|Categorical columns in the dataset are identified. <br> **`LabelEncoder`** is used to convert them to numerical values, assigning a number to each category.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/3.EncodingCategoricalVariables.png" width="4000"/>|
|**`X`** contains all columns except **`'SalePrice'`** (input features). <br> **`y`** contains only **`'SalePrice'`**, which is the variable to be predicted.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/4.SeparaVariables.png" width="4000"/>|
|**`RobustScaler()`** is used to normalize the data and make it more resistant to outliers.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/5.ScalingFeatures.png" width="4000"/>|
|The dataset is split into **80% training** and **20% testing sets**. **`random_state=42`** ensures the split is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/6.SplittingDataset.png" width="4000"/>|
|Import the train_test_split function from the sklearn.model_selection module. This function is used to split a dataset into two parts: <br> Training data (X_train, y_train) → For training the model. <br> Testing data (X_test, y_test) → For evaluating the model.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/7.TrainingLinearRegression.png" width="4000"/>|
|Predictions are generated for both training and testing sets.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/8.GeneratingPredictions.png" width="4000"/>|
|**RMSE (Root Mean Squared Error):** Measures the prediction error. <br> **R² (Coefficient of Determination):** Measures how well the model fits the data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/WithModels/2.DecisionTreeRegressor/Images/9.EvaluatingModel.png" width="4000"/>|