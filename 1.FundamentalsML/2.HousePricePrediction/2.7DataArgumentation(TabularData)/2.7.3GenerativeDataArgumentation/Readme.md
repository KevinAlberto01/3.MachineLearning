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


<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

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
|| <img src = "" width="4000"/>|