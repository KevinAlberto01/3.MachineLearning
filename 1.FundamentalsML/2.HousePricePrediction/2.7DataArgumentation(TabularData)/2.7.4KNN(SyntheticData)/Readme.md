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
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/ComparationData.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/ComparationData(Jittering).png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Graph </h4>
</p>

|Graph (without Jittering)| Graph (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/ComparationGraph.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.4KNN(SyntheticData)/Images/ComparationGraph(Jittering).png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Jittering (Perturbation Leve)) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|| <img src = "" width="4000"/>|