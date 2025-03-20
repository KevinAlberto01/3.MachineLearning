<p align = "center" >
    <h1 align = "Center"> Jittering (Optional)</h1>
</p>

Applies Data Augmentation on a housing price data set by adding Gaussian noise to the numeric variables. It then saves the augmented data set to a CSV file and displays a comparison between the original and modified values.

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

Create an augmented version of the original data set by adding a small amount of Gaussian noise to the numerical variables. This can help improve the robustness of Machine Learning models by making them less sensitive to minor variations in the data.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

|Pseudocode| Image of the program|
|----------|---------------------|
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/FirstDate(Jitters).png" width="4000"/>|

|Pseudocode| Image of the program|
|----------|---------------------|
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/OriginalAugmented(Jitters).png" width="4000"/>|


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Jitters) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/OriginalAugmented(Jitters).png" width="4000"/>|

<p align = "center" >
    <h1 align = "Center"> Noising(Gaussian)</h1>
</p>

This script implements Data Augmentation on tabular data by adding Gaussian noise to the dataset features. It then trains and evaluates four regression models to predict housing prices. Finally, it saves the results in a CSV file and generates comparative graphs of the models' performance.

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

Apply Data Augmentation techniques (Gaussian noise) to increase model robustness and evaluate how it affects the performance of regression models in housing price prediction.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Data </h4>
</p>

|Data (without Jittering)| Data (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/ComparationData.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/ComparationData(Jitters).png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Graph </h4>
</p>

|Graph (without Jittering)| Graph (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/ComparationGraph.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/ComparationGraph(Jitters).png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Noising Gaussian) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|The necessary libraries for data manipulation (**pandas**, **numpy**), visualization (**matplotlib**), preprocessing (**scikit-learn**), model training and metrics evaluation are imported.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/1.ImportLibraries.png" width="4000"/>|
|The dataset containing information about the dwellings is loaded. Then, the column names are printed to verify that the data have been loaded correctly.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/2.LoadData.png" width="4000"/>|
|Three new features are created: <br> **`TotalBathrooms`:** Sum of full and half bathrooms (half bath = 0.5). <br> **`HouseAge`:** Years from house construction to 2025. <br> **`PricePerSF`:** Price per square foot (**`saleprice`** / **`gr_liv_area`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/3.FeatureEngineering.png" width="4000"/>|
|Categorical variables are eliminated and converted into dummy variables using **`pd.get_dummies()`**.<br> We define **`X`** as the independent variables (features) and **`y`** as the target variable (**`saleprice`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/4.DataPreparationModel.png" width="4000"/>|
|RobustScaler is applied to normalize the data, reducing the impact of outliers.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/5.DataNormalization.png" width="4000"/>|
|**Gaussian noise** is added to each column to simulate variations in the data without changing its original structure.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/6.ApplyingGaussian.png" width="4000"/>|
|The data is divided into training (**80%**) and test (**20%**) using **`train_test_split()`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/7.DivisionIntoTraining.png" width="4000"/>|
|Four models are defined: <br> **Linear Regression** <br> **Decision Tree** <br> **Random Forest** <br> **K-Nearest Neighbors (KNN)** | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/8.DefineMachineLearning.png" width="4000"/>|
|For each model: <br> **`X_train`** and **`y_train are trained`.** <br> Predictions are made in training and testing. <br> Metrics are calculated: <br> **RMSE (Root Mean Square Error):** How far the predictions are from the true value. <br> **R² (Coefficient of determination):** What percentage of the variability of **`y`** explain the predictions.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/9.TrainingEvaluation.png" width="4000"/>|
|The results are saved in a CSV file and printed on the console.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/10.SavingResults.png" width="4000"/>|
|**Comparison of RMSE** between training and test. <br> **Comparison of R²** between training and test.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.1Noising(Gaussian)/Images/Noising(Gaussian)/11.MetricVisualization.png" width="4000"/>|