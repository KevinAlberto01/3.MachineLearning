<p align = "center" >
    <h1 align = "Center"> Training Multiple Algorithms</h1>
</p>

Evaluates different Machine Learning models to predict the price of houses using the AmesHousing_cleaned.csv dataset. Feature Engineering techniques, data scaling and training of multiple regression algorithms are applied to compare their performance with error metrics and coefficient of determination (R²).


<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

1. **Dataset loading** and column visualization.
2. **Manual Feature Engineering** (creation of new variables useful to improve the model).
3. **Preprocessing:** Transformation of categorical variables and data scaling.
4. **Division of data** into training and test sets.
5. Definition of **regression models**:
    * **Linear Regression**
    * **Decision Tree**
    * **Random Forest**
    * **K-Nearest Neighbors (KNN)**
6. **Training and evaluation** of each model using metrics such as RMSE, R², MAE and MSE.
7. **Comparison of results** and storage of metrics in a CSV file.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Columns </h4>
</p>
This dataset is a clean version of the Ames Housing Dataset, which contains information about homes in Ames, Iowa, with many characteristics about each property. 
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/Index.png" width="4000"/>

<p align = "center" >
    <h4 align = "Center"> Results summary </h4>
</p>

This is the summary of regression model evaluation in your code.
Each row represents a model (Linear Regression, Decision Tree, Random Forest and KNN) and the columns show various performance metrics.

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/Results.png" width="4000"/>|

**1. Linear Regression:**
- **Test RMSE high (23.733)** indicates that the model does not predict as accurately on new data.
- **Train R² = 0.98 vs. Test R² = 0.93** → There is a small difference, but the model generalizes well.
- **May be affected by nonlinear data.**

**2. Decision Tree:**
- **Train RMSE = 0** and **Train R² = 1** → **Total overfitting** (model memorized training data).
- **Test RMSE = 19.849** and **Test R² = 0.95** → Does not generalize as badly, but could be optimized with pruning or adjustments.

**3.Random Forest (Best model):**
- **Lower RMSE Test (16.536)** → Better accuracy on new data.
- **Test R² = 0.96** → Explains variability in test data well.
- **Good balance between training and testing (less overfitting than decision tree).**

**4.KNN (Worst model):**
- **Test RMSE very high (47.591)** → Poor predictions on new data.
- **Test R² = 0.71** → Explains only 71% of price variability.
- **Problem: KNN does not handle data with many dimensions or complex relationships well.**

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** For data manipulation. <br> **numpy:** For numerical calculations. <br> **train_test_split:** For splitting data into training and testing. <br> **RobustScaler:** For scaling data and reducing the impact of outliers. <br> **LinearRegression**, **DecisionTreeRegressor**, **RandomForestRegressor**, **KNeighborsRegressor**: Regression models to evaluate. <br> **mean_squared_error**, **r2_score**, **mean_absolute_error:** Model evaluation metrics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F1.ImportLibraries.png" width="4000"/>|
|The CSV file is loaded into a DataFrame **`df`**. <br> The available columns in the dataset are printed.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F2.LoadDataset.png" width="4000"/>|
|**Three new features** are created: <br> **TotalBathrooms:** Sum of full bathrooms and half bathrooms (half bath is considered as 0.5). <br> **houseage:** Age of the house in the year 2025. <br> **PricePerSF:** Price per square foot (**`saleprice / gr_liv_area`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F3.ManualFeature.png" width="4000"/>|
|**`saleprice`** is the target variable **`y`** (what we want to predict). <br> Unnecessary columns are eliminated and categorical variables are converted to numeric with **`pd.get_dummies()`**. <br> **RobustScaler** is applied, which is useful to **reduce the impact of outliers**. <br> The data is divided into **80%** training and **20%** testing.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F4.DataPreprocessing.png" width="4000"/>|
**Four models** are defined: <br> **`Linear Regression`:** Basic model without regularization. <br> **`DecisionTreeRegressor`:** Non-linear model that can capture more complex relationships. <br> **`Random Forest (RandomForestRegressor)`:** Set of multiple decision trees to improve accuracy. <br> **`K-Nearest Neighbors (KNN)`:** Model based on nearest neighbors.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F5.DefineModels.png" width="4000"/>|
|Each model is trained with the training data. <br> Predictions are generated for both training and testing. <br> The following evaluation **metrics are calculated:** <br> **RMSE (Root Mean Squared Error):** Root mean squared error. <br> **R² (Coefficient of Determination):** Indicates how well the model explains the variability of the data. <br> **MAE (Mean Absolute Error):** Mean Absolute Error. <br> **MSE (Mean Squared Error):** Mean Squared Error. <br> The results are stored in a **`results list`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F6.ModelTrainingEvaluation.png" width="4000"/>|
|The **`results`** list is converted into a **`DataFrame`** for better visualization. <br> The results are saved in a CSV file for future reference.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.6FeatureEngineeringManual/Images/F7.SavingDisplayingResults.png" width="4000"/>|
