<p align = "center" >
    <h1 align = "Center"> 3.Random Forest Regressor</h1>
</p>

Implements a K-Nearest Neighbors Regressor (KNN) model to predict housing prices using the Ames Housing dataset. Feature engineering techniques, data scaling and data augmentation with Gaussian noise are applied to improve the robustness of the model.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/WithModels/3.RandomForestRegressor/Images/Data.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas** → To handle data in a DataFrame. <br> **numpy** → For numerical calculations and noise generation (jittering). <br> **train_test_split** → To split the data into training and test sets. <br> **RobustScaler** → To scale the data and reduce the influence of outliers. <br> **RandomForestRegressor** → A model based on multiple decision trees.  <br> **mean_squared_error**, **r2_score** → Metrics to evaluate the model.|<img src = "" width="4000"/>|
|Add random noise to the data (**`X`**) to improve the model's generalization. <br> Functioning: <br> For each column, generate a random perturbation between **`-0.01`** and **`0.01`** multiplied by the standard deviation of the column. <br> Add this perturbation to the original values of the columns <br> Jittering can help in models like linear regression, but in decision trees, its impact is more uncertain, as these models are more robust to small variations in the data.|<img src = "" width="4000"/>|
|The dataset **`AmesHousing_cleaned.csv`** is loaded into a **`DataFrame`**.|<img src = "" width="4000"/>|
|**`TotalBathrooms:`** Sum **`full_bath`** and half of **`half_bath`**. <br> **`HouseAge`:** Subtract the year of construction from **`2025`** to get the age of the house. <br> **`PricePerSF`:** Calculates the price per square foot (**`saleprice`** / **`gr_liv_area`**).|<img src = "" width="4000"/>|
|**`X` (features):** The **`saleprice`** (target variable) was removed and categorical variables were converted to numerical using **`pd.get_dummies()`**. <br> **`y` (target):** **`saleprice`** is assigned. <br> **`RobustScaler()`:** Scales the features to reduce the impact of outliers. <br> **Apply Jittering** to the scaled data.|<img src = "" width="4000"/>|
|**`test_size=0.2`** → 80% training, 20% test. <br> **`random_state=42`** → To obtain reproducible results.|<img src = "" width="4000"/>|
|**Random Forest** is a set of multiple decision trees. <br> Each tree is trained with a **random part** of the data. <br >Averages the results of all the trees to make the prediction.|<img src = "" width="4000"/>|
|**Predictions** are made for **`X_train`** and **`X_test`**.|<img src = "" width="4000"/>|
|**RMSE (Root Mean Square Error):** <br> **`mean_squared_error()`** calculates the mean squared error. <br> **`np.sqrt()`** takes the square root to obtain the error in the same units as **`saleprice`**. <br> **R² (Coefficient of determination):** <br> **`r2_score()`** measures how well the model explains the variance in the data.|<img src = "" width="4000"/>|
|The **RMSE** and **R²** metrics are printed for the training and test sets.|<img src = "" width="4000"/>|