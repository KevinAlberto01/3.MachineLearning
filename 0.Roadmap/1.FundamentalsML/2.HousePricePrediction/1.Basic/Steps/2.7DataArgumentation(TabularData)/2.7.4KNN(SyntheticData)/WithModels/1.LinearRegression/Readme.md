<p align = "center" >
    <h1 align = "Center"> 1.Lineal Regression</h1>
</p>

This program implements a Linear Regression model to predict housing prices using the AmesHousing dataset. Data preprocessing including feature engineering, normalization and data augmentation with jittering is performed to improve the robustness of the model.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/WithModels/1.LinearRegression/Images/Data.png" width="4000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Lineal Regression) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** To load and manipulate tabular data (DataFrames). <br> **numpy:** To handle numeric arrays and calculations. <br> **train_test_split:** To split data into training and test sets. <br> **RobustScaler:** To scale the data and reduce the effect of outliers. <br> **LinearRegression:** To train a linear regression model. <br> **mean_squared_error, r2_score:** To evaluate model performance. <br>**NearestNeighbors:** To search for nearest neighbors in the data and generate synthetic data with KNN.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/WithModels/1.LinearRegression/Images/1.ImportLibraries.png" width="4000"/>|
|It uses **K-Nearest Neighbors (KNN)** to find the nearest neighbors of each point in the original data. <br> For each point, it **creates a new synthetic point** by interpolating between the **original point and one of its neighbors**. <br> It returns an **extended** dataset (X and y with synthetic data added).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/WithModels/1.LinearRegression/Images/2.SyntheticDataGeneration.png" width="4000"/>|
|It creates new columns that can be useful to improve the model prediction. <br> **`pd.get_dummies()`:** Converts categorical variables to numeric variables (one-hot encoding). <br> **`drop_first=True:`** Avoids multicollinearity by dropping one category per categorical variable. <br> **`saleprice`** is the variable we want to predict.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/WithModels/1.LinearRegression/Images/3.DatasetLoading.png" width="4000"/>|
|Less sensitive to outliers than **`StandardScaler`** or **`MinMaxScaler`**. <br> Transforms data by subtracting the **median** and **scaling** by the interquartile range (IQR).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/WithModels/1.LinearRegression/Images/4.ScalingGeneration.png" width="4000"/>|
|Divide the dataset into **80% training** and **20% testing**. <br> **`random_state=42:`** Ensures that the results are reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/WithModels/1.LinearRegression/Images/5.TrainTestSeparation.png" width="4000"/>|
|Train a Linear Regression model using the training data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/WithModels/1.LinearRegression/Images/6.ModelTraining.png" width="4000"/>|
| Makes predictions on training and test data. <br> **`RMSE (Root Mean Squared Error)`:** Measures the average error in the prediction (the smaller, the better). <br> **`R² Score`:** Measures the proportion of variability explained by the model (the closer to 1, the better). <br> It shows the results of the model evaluation.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/WithModels/1.LinearRegression/Images/7.ModelPrediction.png" width="4000"/>|