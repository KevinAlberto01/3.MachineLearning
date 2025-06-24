<p align = "center" >
    <h1 align = "Center"> Training Multiple Algorithms</h1>
</p>

various regression algorithms are trained and compared to predict the price of a house. 
Different Machine Learning models are tested to evaluate which one performs best according to various error and accuracy metrics.

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

* **1.Data Loading and Preparation**
    - The clean dataset (**`AmesHousing_cleaned.csv`**) is loaded.
    - The features (**`X`**) and the target variable (**`y = saleprice`**) are separated.
    - The categorical columns are identified and **One-Hot Encoding** is applied to convert them into numerical variables.
    - The data is divided into a **training set** (**80%**) and a **test set** (**20%**).
    - Normalization is applied with **`StandardScaler`** to scale the features.

* **2.Model Selection and Optimization**
    - A function **`get_best_decision_tree()`** is defined that uses **GridSearchCV** to find the best **`max_depth`** in a **Decision Tree**.
    - Four models are stored in a dictionary:
        - **Linear Regression (`LinearRegression`)**
        - **Decision Tree (`DecisionTreeRegressor`)** → Optimized with GridSearchCV.
        - **Random Forest (`RandomForestRegressor`)**
        - **K-Nearest Neighbors (`KNeighborsRegressor`)**

* **3.Model Training and Evaluation**
    - Each model is trained and performance metrics are calculated:
        - **RMSE (Root Mean Squared Error)**
        - **Puntuación R²**
        - **MAE (Mean Absolute Error)**
        - **Explanation of Variance**
    - The results of each model are printed to the console.
    - The results are stored in a DataFrame and saved in a CSV file.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Description </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/1.LoadingCleanedData.png" width="4000"/>

<p align = "center" >
    <h4 align = "Center"> Description </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/2.ShapeXafterOne-HotEncoding.png" width="4000"/>

<p align = "center" >
    <h4 align = "Center"> Description </h4>
</p>

Detected Categorical Columns: ['ms_zoning', 'street', 'alley', 'lot_shape', 'land_contour', 'utilities', 'lot_config', 'land_slope', 'neighborhood', 'condition_1', 'condition_2', 'bldg_type', 'house_style', 'roof_style', 'roof_matl', 'exterior_1st', 'exterior_2nd', 'mas_vnr_type', 'exter_qual', 'exter_cond', 'foundation', 'bsmt_qual', 'bsmt_cond', 'bsmt_exposure', 'bsmtfin_type_1', 'bsmtfin_type_2', 'heating', 'heating_qc', 'central_air', 'electrical', 'kitchen_qual', 'functional', 'fireplace_qu', 'garage_type', 'garage_finish', 'garage_qual', 'garage_cond', 'paved_drive', 'pool_qc', 'fence', 'misc_feature', 'sale_type', 'sale_condition']

<p align = "center" >
    <h4 align = "Center"> Shape of x after One-Hot encoding </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/3.Description" width="4000"/>

<p align = "center" >
    <h4 align = "Center"> Best max_depth for DecisionTreeRegressor </h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/4.BestMaxDepth.png" width="4000"/>

<p align = "center" >
    <h4 align = "Center"> Comparation </h4>
</p>

|Linear Regression |Decision Tree Regressor | Random Forest Regressor| K-Nearest Neighbors Regressor|
|-----------------------------------|------------------------|---------------------|--------------|
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/5.LinealRegression.png" width="4000"/> |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/6.DEcisionTreeRegressor.png" width="4000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/7.RandomForestRegressor.png" width="4000"/> |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.3TraininigMultipleAlgorithms/Images/8.KNeighborsRegressor.png" width="4000"/>|

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