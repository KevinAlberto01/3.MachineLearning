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
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/Data(Jittering).png" width="4000"/>|

|Pseudocode| Image of the program|
|----------|---------------------|
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/Comparation(Jittering).png" width="4000"/>|


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Jitters) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas** is used to handle data structures such as **DataFrame**. <br>**numpy** is used for mathematical operations and random noise generation.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/L1.ImportLibraries.png" width="4000"/>|
|The **`AmesHousing_cleaned.csv`** dataset is loaded into a pandas **`DataFrame`**.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/L2.DataLoading.png" width="4000"/>|
|A summary statistic of the data is printed before any processing. <br> **`df.describe()`** displays statistics such as mean, standard deviation, minimum and maximum values of the numeric columns.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/L3.InitialDataSummary.png" width="4000"/>|
|Only the numeric columns of the **`DataFrame`** are selected, since the perturbation will only be applied to numeric values.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/L4.SelectionNumericColumns.png" width="4000"/>|
|**`add_jitter`** is a function that adds noise to the data to simulate variations and improve the generalization of the model. <br> **`np.random.normal(loc=0.0, scale=noise_level * df[col].std(), size=len(df))`** generates random noise with mean **`0.0`** and a standard deviation proportional to the column. <br> The noise is added to each numeric column.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/L5.DefinitionJitteringFunction.png" width="4000"/>|
|The add_jitter function with a noise level of **`0.02`** is applied.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/L6.ApplicationJittering.png" width="4000"/>|
|The first 5 values of the **`saleprice`** column before and after jittering are compared.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/L7.ComparisonOA.png" width="4000"/>|
|The augmented **`DataFrame`** is saved in a CSV file without the index.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/L8.SavingAugmentedDataset.png" width="4000"/>|
|A success message is printed indicating that the transformation is complete. <br> The number of rows before and after the transformation is displayed (they should be equal, since only noise was applied and no new rows were generated).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/L9.FinalMessages.png" width="4000"/>|

<p align = "center" >
    <h1 align = "Center"> Jittering Perturbation Leve </h1>
</p>

This program is to compare the performance of different Machine Learning models for predicting house prices using the AmesHousing_cleaned.csv dataset. In addition, jittering is applied as a Data Augmentation technique to improve the generalization capability of the models.

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

**1. Data Load:**
* A CSV file containing information about various home features and their selling price is read.

**2. Feature Engineering:**
* New features are created to improve model performance, such as:
    * **TotalBathrooms:** Total number of bathrooms.
    * **HouseAge:** Age of the house (2025 - year of construction).
    * **PricePerSF:** Price per square foot.

**3. Data Preprocessing:**
* Categorical variables are converted to one-hot encoding.
* Normalized features with RobustScaler to reduce the impact of outliers.

**4.Data Augmentation with Jittering:**
* Small random noise is added to the data to avoid overfitting and improve the model's ability to generalize on new data.

**5.Split into Training and Test Set:**
* Data is separated into 80% training and 20% testing.

**6.Model Training:**
* Four regression models are evaluated:
    * Linear Regression
    * Decision Tree
    * Random Forest
    * K-Nearest Neighbors (KNN)
*   Each model is trained and evaluated using RMSE (Root Mean Square Error) and R² Score (Coefficient of Determination).

**7.Storage and Visualization of Results:**
* Results are stored in a CSV file.
* RMSE and R² values are plotted to compare model performance.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Data </h4>
</p>

|Data (without Jittering)| Data (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/ComparationData.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/ComparationData(Jittering).png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Graph </h4>
</p>

|Graph (without Jittering)| Graph (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/ComparationGraph.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/ComparationGraph(Jittering).png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Jittering (Perturbation Leve)) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas**, **numpy** for data manipulation. <br> **matplotlib.pyplot** for visualization of results. <br> **train_test_split** to split the data into training and test. <br> **RobustScaler** to scale the data and reduce the influence of outliers. <br> Models: <br> **`LinearRegression`** (**Linear Regression**). <br> **`DecisionTreeRegressor`** (**Decision Tree**) <br> **`RandomForestRegressor`** (**Random Forest**) <br> **`KNeighborsRegressor`** (**K-Nearest Neighbors**) <br> **mean_squared_error**, **r2_score** to evaluate model performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/1.ImportLibraries.png" width="4000"/>|
|Jittering helps reduce overfitting by making the model less dependent on small fluctuations in the data. <br> This function adds **random noise (jittering)** to the data to improve model generalization. <br> **`perturbation_level=0.01`**: the noise level is 1% of the standard deviation of each column. <br> A **uniform** perturbation between **`-0.01 * std`** and **`0.01 * std`** is generated for each column value.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/2.DefinitionFunction.png" width="4000"/>|
|The dataset **AmesHousing_cleaned.csv** is loaded, which contains house information with variables such as: <br> Year of construction, house size, number of bathrooms, etc. <br> Sale price (**`saleprice`**) as target variable.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/3.LoadingDataset.png" width="4000"/>|
|New features are created to improve the quality of the dataset: <br> **TotalBathrooms:** Sum the full bathrooms and half of the half bathrooms. <br> **HouseAge:** Calculates the age of the house based on the current year (2025). <br> **PricePerSF:** Price per square foot (saleprice / gr_liv_area).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/4.FeatureEngineering.png" width="4000"/>|
|Separate **X** (features) and **y** (target = house price). <br> Convert categorical variables to **one-hot encoding** with **`pd.get_dummies()`**, removing one category (**`drop_first=True`**) to avoid collinearity. <br> Normalize **X** with **`RobustScaler`**, which **reduces the impact of outliers.** <br> **Jittering** is applied to **`X_scaled`** to add artificial variations to the data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/5.DataPreparation.png" width="4000"/>|
|It is divided into **80% training** and **20% test** (**`test_size=0.2`**). <br> **`Random_state=42`** is used to obtain reproducible results.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/6.DivisionIntoTraining.png" width="4000"/>|
|A dictionary is defined with the models to be evaluated. <br> Each model is trained and evaluated: <br> It is fitted (**`.fit()`**) with the training data. <br> **Train** and **test** predictions are made. <br> Metrics are calculated: <br> **RMSE:** Root mean square error (average error in prediction). <br> **R² Score:** Explains what percentage of the variance of the data explains the model.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/7.TrainingEvaluation.png" width="4000"/>|
|The results are saved in a CSV file. <br> The results are plotted: <br> **`RMSE:`** Shows how well each model predicts price. <br> **`R² Score`:** Indicates how well the models fit.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.2Jittering(PerturbationLeve)/Images/8.SavingViewing.png" width="4000"/>|
