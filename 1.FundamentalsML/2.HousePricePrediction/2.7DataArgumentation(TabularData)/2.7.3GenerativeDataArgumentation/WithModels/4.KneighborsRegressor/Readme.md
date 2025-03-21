<p align = "center" >
    <h1 align = "Center"> 4.Kneighbors Regressor</h1>
</p>

Implements a K-Nearest Neighbors Regressor (KNN) model to predict housing prices using the Ames Housing dataset. Feature engineering techniques, data scaling and data augmentation with Gaussian noise are applied to improve the robustness of the model.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
numerical calculations and noise generation (jittering). <br> **train_test_split** → To split the data into training and test sets. <br> **RobustScaler** → To scale the data and reduce the influence of outliers. <br>Import the **KNeighborsRegressor** class from sklearn.neighbors.<br> **KNeighborsRegressor** is a model based on the K-Nearest Neighbors (KNN) algorithm, which makes predictions by finding the K nearest neighbors to a given point and averaging their values.<br> **mean_squared_error**, **r2_score** → Metrics to evaluate the model.|<img src = "1.ImportLibraries.png" width="4000"/>|Add random noise to the data (**`X`**) to improve the model's generalization. <br> Functioning: <br> For each column, generate a random perturbation between **`-0.01`** and **`0.01`** multiplied by the standard deviation of the column. <br> Add this perturbation to the original values of the columns <br> Jittering can help in models like linear regression, but in decision trees, its impact is more uncertain, as these models are more robust to small variations in the data.|<img src = "" width="4000"/>|
|The dataset **`AmesHousing_cleaned.csv`** is loaded into a **`DataFrame`**.|<img src = ""/>|
|**`TotalBathrooms:`** Sum **`full_bath`** and half of **`half_bath`**. <br> **`HouseAge`:** Subtract the year of construction from **`2025`** to get the age of the house. <br> **`PricePerSF`:** Calculates the price per square foot (**`saleprice`** / **`gr_liv_area`**).|<img src = "" width="4000"/>|
|**`X` (features):** The **`saleprice`** (target variable) was removed and categorical variables were converted to numerical using **`pd.get_dummies()`**. <br> **`y` (target):** **`saleprice`** is assigned. <br> **`RobustScaler()`:** Scales the features to reduce the impact of outliers. <br> **Apply Jittering** to the scaled data.|<img src = "" width="4000"/>|
|**`test_size=0.2`** → 80% training, 20% test. <br> **`random_state=42`** → To obtain reproducible results.|<img src = "" width="4000"/>|
|Create a KNN regression model with the default values. The model will search for the 5 nearest neighbors (K=5 by default) and take the average of their target values to predict a new data point.|<img src = "" width="4000"/>|
|**Predictions** are made for **`X_train`** and **`X_test`**.|<img src = "" width="4000"/>|
|**RMSE (Root Mean Square Error):** <br> **`mean_squared_error()`** calculates the mean squared error. <br> **`np.sqrt()`** takes the square root to obtain the error in the same units as **`saleprice`**. <br> **R² (Coefficient of determination):** <br> **`r2_score()`** measures how well the model explains the variance in the data.|<img src = "" width="4000"/>|
|The **RMSE** and **R²** metrics are printed for the training and test sets.|<img src = "" width="4000"/>|
