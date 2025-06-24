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
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/Data.png" width="4000"/>|

|Pseudocode| Image of the program|
|----------|---------------------|
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/Information.png" width="4000"/>|


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Jitters) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** It is used to handle tabular data. <br> **numpy:** It allows generating random numbers and performing mathematical operations. <br> **sklearn.preprocessing.StandardScaler:** It is used to scale the data before applying jittering and then restoring it.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J1.ImportLibraries.png" width="4000"/>|
|The CSV file containing the house data is read. <br> **`df`** is a pandas DataFrame that contains all the columns and rows of the dataset.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J2.LoadData.png" width="4000"/>|
|**`df.describe()`** shows general statistics of the dataset: <br> **Mean** (**`mean`**) <br> **Minimum** (**`min`**) and **Maximum**(**`max`**) <br> **Standard deviation** (**`std`**) <br> Percentiles (25%, 50%, 75%) <br> This helps to understand the distribution of numerical values.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J3.StaticalSummary.png" width="4000"/>|
|**`df.select_dtypes(include=[np.number])`:** <br> Select only the columns that contain numerical data. <br> **`df_numeric`:** <br> It's a new DataFrame that contains only the numeric values.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J4.NumericalVariables.png" width="4000"/>|
|Different columns can have values with very different scales (e.g., prices in thousands and sizes in square meters). <br> **`StandardScaler()`** transforms the values so that they have: <br> **Media = 0** <br> **Standard deviation = 1** <br> **`fit_transform(df_numeric)`:** <br> Adjust the scaler to the data and transform them to the standard scale. <br> **`X_scaled`:** <br> It contains the scaled numerical values.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J5.ScaleData.png" width="4000"/>|
|It is a technique that introduces small variations in the data to **increase the diversity** of the dataset. <br> The function **`add_jitter()`** is defined, which: <br> Generates random noise with **`np.random.normal()`**, which follows a **normal distribution** with: <br> **`loc=0.0`:** Mean of the noise is 0 (does not change the mean value of the data). <br> **`scale=noise_level`:** Noise magnitude is 2% of the original value (**`0.02`**). <br> The noise is added to /**`X_scaled`**, creating **`X_jittered`**.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J6.ApplyJittering.png" width="4000"/>|
|The jittering was applied in standardized values. <br> To make the data interpretable, the transformation is inverted with **`scaler.inverse_transform()`**. <br> **`df_jittered`:** <br> DataFrame with the numeric columns, but with values augmented by jittering.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J7.RestoreValues.png" width="4000"/>|
|**`df.select_dtypes(exclude=[np.number])`:** <br> Filters only **categorical variables** (text, categories, etc.). <br> **`df_categorical`:** <br> Contains only the categorical columns of the original dataset. <br> We can't apply jittering to text, so random values are chosen from the original categories. <br> **`df_categorical.sample(n=df_jittered.shape[0], replace=True)`:** <br> Takes a random sample of the existing categories. <br> **`replace=True`:** Allows some values to be repeated. <br> **`random_state=42`:** Used to ensure that the sampling is reproducible.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J8.HandlingCategorical.png" width="4000"/>|
|Combines the numeric data (**`df_jittered`**) with the categorical variables (**`df_categorical_jittered`***). <br> **`reset_index(drop=True)`:** Ensures that the indexes are aligned correctly.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J9.CombineNumericalCategorical.png" width="4000"/>|
|To be able to use the augmented data in training machine learning models. <br> **`to_csv(output_path, index=False)`:** <br> Saves the DataFrame without the indexes, only with the original columns.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J10.AugmentedDataset.png" width="4000"/>|
|Confirmation message indicating that jittering has been performed successfully. <br> The number of rows in the original dataset and the augmented dataset are displayed (they should be equal, since jittering does not generate new rows, it only modifies the existing values).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J11.FinalMessages.png" width="4000"/>|

<p align = "center" >
    <h1 align = "Center"> Generative Data Argumentation </h1>
</p>

This code implements a complete Machine Learning workflow to predict house prices using regression methods. We work with a dataset that has been augmented using Jittering (Data Augmentation for tabular data).

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

* Predict house prices using Machine Learning models.
* Compare the performance of different regression algorithms. 
* Apply Feature Engineering techniques to improve the model. 
* Use Data Augmentation (Jittering) to generate more data and improve generalization.
* Evaluate the models with metrics such as RMSE and R². 
* Save and visualize results for later analysis.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Data </h4>
</p>

|Data (without Jittering)| Data (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/ComparationData.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/ComparationData(Jittering).png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Graph </h4>
</p>

|Graph (without Jittering)| Graph (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/ComparationGraph.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/ComparationGraph(Jittering).png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Jittering (Perturbation Leve)) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas:** Loading and manipulating tabular data. <br> **numpy:** Mathematical calculations, such as square root for RMSE. <br> **matplotlib.pyplot:** Create plots to compare models. <br> **sklearn.model_selection.train_test_split:** Split data into training and testing. <br> **sklearn.preprocessing.RobustScaler:** Scale data robustly against outliers. <br> **sklearn.preprocessing.LabelEncoder:** Convert categorical variables into numbers. <br> **sklearn models:** Linear Regression, Decision Tree, Random Forest, KNN. <br> **sklearn.metrics:** Calculate RMSE and R² to evaluate models.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G1.ImportLibraries.png" width="4000"/>|
|The CSV file is read with the data that has already been augmented using **jittering**. <br> **`df`** contains all the columns of the dataset.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G2.LoadingDataset.png" width="4000"/>|
|**New features** derived from existing columns are created: <br> **`TotalBathrooms`:** <br> Full and half full bathrooms and half half half bathrooms are added. <br> **`HouseAge:`** <br> The age of the house is calculated by subtracting the year of construction to 2025. <br> **`PricePerSF`:** <br> Price per square foot of the house (saleprice divided by gr_liv_area).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G3.FeatureEnginnering.png" width="4000"/>|
|**Identifies categorical columns** (text or categories). <br> **Convert categories to numbers** using **`LabelEncoder()`**. <br> Machine learning models cannot process text directly, they need numbers.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G4.CodingCategorical.png" width="4000"/>|
|**`X`:** All columns except **`saleprice`** (predictor variables). <br> **`y`**: The column **`saleprice`**, which is the target variable we want to predict.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G5.DefiningVariables.png" width="4000"/>|
|It is robust to **outliers**, unlike **`StandardScaler()`**. <br> It takes care of normalizing the values, reducing the influence of extreme values. <br> **`fit_transform(X)`:** <br> Fits and transforms data in a single operation.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G6.ScalingNumerical.png" width="4000"/>|
|Divide the data into: <br> **80%** training (**`X_train`**, **`y_train`**). <br> **`20%`** test (**`X_test`**, **`y_test`**). <br> **`random_state=42`:** Ensures that the division is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G7.DivideTrainingTest.png" width="4000"/>|
|**Linear Regression** (**`LinearRegression`**) <br> A simple model that assumes a linear relationship between variables. <br> **DecisionTreeRegressor** (**`DecisionTreeRegressor`**) <br> A nonlinear model that divides data into decision nodes. <br> **Random Forest** (**`RandomForestRegressor`**) <br> Several decision trees combined to improve prediction. <br> **K-Nearest Neighbors** (**`KNeighborsRegressor`**) <br> Predicts the value based on the K nearest neighbors.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G8.DefineModels.png" width="4000"/>|
|Each model is trained with **`model.fit(X_train, y_train)`** <br> **Predictions** are made on **`X_train`** and **`X_test`**. <br> Evaluation metrics are calculated: <br> **RMSE (Root Mean Squared Error):** Root Mean Squared Error. <br> **R² Score:** How well the model explains the variability of the data. <br> The results are stored in a **`results[]`** list.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G9.TrainEvaluate.png" width="4000"/>|
|**`Results`** are converted into a **DataFrame** and saved in a CSV file.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G10.ResultsCSV.png" width="4000"/>|
|A **bar chart** is created to compare the RMSE of the models.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/G11.Graphical.png" width="4000"/>|