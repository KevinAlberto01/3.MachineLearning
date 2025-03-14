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
**Pandas and NumPy:** For data handling and processing. <br> **Matplotlib:** For visualizations (although it is not used in this code). <br> **sklearn.model_selection:** To split the dataset into training and testing, as well as to perform Grid Search to optimize hyperparameters. <br> **sklearn.linear_model, sklearn.tree, sklearn.ensemble, sklearn.neighbors:** Regression models to evaluate. <br> **sklearn.metrics:** To measure the performance of the models. <br> **Sklearn.preprocessing:** For scaling the data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.1ImportLibraries.png" width="4000"/>|
The dataset **`AmesHousing_cleaned.csv`** is loaded, which is a clean version of the Ames Housing dataset.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.2LodingDataset.png" width="4000"/>|
**`X`** contains all the columns except **`saleprice`** (the features). <br> **`y`** contains the target variable **`saleprice`** (the house prices).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.3SeparateFeatured.png" width="4000"/>|
Categorical columns are identified. <br> **One-Hot Encoding is applied**, transforming each category into a binary column. <br> **`drop_first=True`** is used to avoid collinearity (the first category of each variable is removed)| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.4TransformationCategorical.png" width="4000"/>|
**80% training data**, **20% test data**. <br> **`random_state=42`** ensures reproducibility.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.5.DivisionIntoTrainingTestSets.png" width="4000"/>|
**StandardScaler** is applied to scale the data to a mean of 0 and a standard deviation of 1.🔹  It is fitted on **`X_train`** (**`fit_transform`**) and **transformed** on **`X_test`** (**transform**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.6FeatureNormalization.png" width="4000"/>|
The best **`max_depth`** for **`DecisionTreeRegressor`** is being sought using **GridSearchCV** with **cross-validation (cv=5)**. <br> **`scoring='neg_mean_squared_error'`** indicates that the mean squared **error will be minimized.** <br> An optimized model with the best **`max_depth`** is returned. <br> The best decision tree model is obtained.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.7SearchBetterHyperparameters.png" width="4000"/>|
**Linear Regression**, **Decision Tree (optimized)**, **Random Forest**, and **KNN** models are created.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.8DefinitionModels.png" width="4000"/>|
Each model is **trained** and **evaluated** using various metrics: <br> **RMSE** (Root Mean Square Error). <br> **R²** (Coefficient of determination). <br> **MAE** (Mean Absolute Error). <br> **Explained variance.** <br>All models are being evaluated.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.9ModelEvaluation.png" width="4000"/>|
The results are saved in an ordered CSV.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.10StorageResults.png" width="4000"/>|