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
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/Results.png" width="4000"/>|

|Pseudocode| Image of the program|
|----------|---------------------|
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/Data.png" width="4000"/>|


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Jitters) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas** → To handle and process tabular data. <br> **numpy** → For mathematical operations and random data generation. <br> **NearestNeighbors (from sklearn.neighbors)** → Finds the **K nearest neighbors** of each point to generate synthetic data.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/1.Import necessary libraries.png" width="4000"/>|
|Load the data from a CSV file.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/2.Load the dataset.png" width="4000"/>|
|**`df.describe()`** → Displays general statistics (mean, standard deviation, percentiles, etc.) only for the numerical columns.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/3.DisplayStatisticalSummary.png" width="4000"/>|
|Extract only the numerical columns, as KNN cannot work with categorical data directly.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/4.SelectNumericalColumns.png" width="4000"/>|
|**`k=5`** is defined, meaning each point will consider its **5 nearest neighbors.** <br> The model is fitted with the numerical data.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/5.ConfigureKNN.png" width="4000"/>|
|**`synthetic_data`** is initialized as an empty list to store the new points. <br> Each row (**`point`**) of **`df_numeric`** is iterated over. <br> Its **K nearest neighbors** are found. <br> For each neighbor (except the point itself), a new synthetic point is generated with: <br> **synthetic_point = original_point + random_factor * (neighbor - original_point)** <br> This method maintains the original data distribution. <br> The new point is stored in **`synthetic_data`**.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/6.GenerateSynthetic.png" width="4000"/>|
|Convert the list of synthetic data into a **`DataFrame`** with the same numerical columns.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/7.Create DataFrame.png" width="4000"/>|
|Concatenate the original and synthetic data into a single **`DataFrame`**.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/8.CombineOriginalSynthetic.png" width="4000"/>|
|Extract the categorical columns. <br> For the synthetic data, random rows from the original data are taken to maintain the proportion. <br> Concatenate the augmented numerical data with the original and synthetic categorical data.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/9.AddOriginalCategorical.png" width="4000"/>|
|Save the augmented dataset to a CSV file.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/10.SaveNewDataset.png" width="4000"/>|
|Confirm that the synthetic data generation has completed successfully. <br> Display how many rows the original dataset had and how many the new augmented dataset has.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/11.PrintResults.png" width="4000"/>|

<p align = "center" >
    <h1 align = "Center"> Generative Data Argumentation </h1>
</p>

This code implements a Machine Learning workflow for house price prediction. It utilizes feature engineering techniques and data augmentation via interpolation with K-Nearest Neighbors (KNN). Then, four regression models are trained and compared:

* Linear Regression
* Decision Tree
* Random Forest
* KNN Regressor

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

The purpose of this code is to improve the accuracy of regression models by generating synthetic data with KNN. This allows the models to generalize better and reduce the prediction error in estimating the price of homes.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Data </h4>
</p>

|Data (without Jittering)| Data (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/ComparationData.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/ComparationData(Jittering).png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Graph </h4>
</p>

|Graph (without Jittering)| Graph (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/ComparationGraph.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/ComparationGraph(Jittering).png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 KNN (Systhetic Data) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas** and **numpy:** Data handling. <br> **matplotlib** and **seaborn:** Plotting. <br> **sklearn:** Preprocessing, ML models, and metrics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/K1.ImportingLibraries.png" width="4000"/>|
|**KNN** (**`NearestNeighbors`**) is used to find nearby points in the original data.<br> A real point is chosen, and its nearest neighbors are searched for. <br> Interpolation between the original point and a random neighbor generates new synthetic points. <br> The synthetic data is then added to the original data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/K2.FunctionGenerate.png" width="4000"/>|
|**TotalBathrooms:** Full bathrooms + half bathrooms. <br> **HouseAge:** Age of the house in years. <br> **PricePerSF:** Price per square foot.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/K3.LoadingPreparing.png" width="4000"/>|
|The feature set (**`X`**) and the target variable (**`y`**) are separated. <br> Categorical variables are converted into dummies. <br> **`RobustScaler`** is used to prevent extreme values from affecting training. <br> 500 synthetic points are generated with KNN and combined with the original data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/K4.PreaparationModeling.png" width="4000"/>|
|80% of the data is used for training, and 20% is used for testing.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/K5.SplittingTrainingTesting.png" width="4000"/>|
|Four Machine Learning models are defined: <br> **Linear Regression** <br> **Decision Tree** <br> **Random Forest** <br> **KNN Regressor** <br> Each model is trained and evaluated with: <br> **RMSE (Root Mean Squared Error):** Average prediction error. <br> **R² (Coefficient of Determination)**: How well the model explains the variability of the data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/K6.TrainingModels.png" width="4000"/>|
|The results are saved to a CSV. <br> The models are compared with bar plots: <br> Test RMSE: Which model has the lowest error. <br> Test R²: Which model has the best accuracy.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/K7.SavingVisualizing.png" width="4000"/>|