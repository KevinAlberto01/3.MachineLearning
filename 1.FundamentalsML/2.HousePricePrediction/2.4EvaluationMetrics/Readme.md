<p align = "center" >
    <h1 align = "Center"> Evaluation Metrics</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

Analyze the obtained metrics, detect if there is overfitting or underfitting, compare the models objectively, and select the most suitable model to predict SalePrice.
Analyze in detail the performance of each evaluated model. Its purpose is to compare the key metrics, detect possible issues (such as overfitting or underfitting), and select the model with the best balance between accuracy and error on test data.

* **1.Load of Results** 
    - This program assumes that the DataFrame df_results was generated and saved by Program 3.
    - The recorded metrics for each model are loaded.

* **2.Analysis of Overfitting and Underfitting**
    - For each model, the following is analyzed:
        - Is the training error very low and the test error very high?
            - If that's the case, it's a sign of overfitting (it learned the training data too well and fails to generalize).
        
        - Is the error high both in training and testing?
            - This is a sign of underfitting (too simple a model for the problem).
    - Key Comparison
        - Difference between Train R² and Test R²: If the difference is very large, there is probably overfitting.
        - Difference between Train RMSE and Test RMSE: If the test error is much greater than the train error, there is overfitting.

* **3.Comparison and Selection of the Best Model**
    - The models are ordered by Test RMSE (prioritizing lower error on test data).
    - It is also checked which one has the highest Test R².
    - The best model will be the one that achieves:
        - Lower Test RMSE 
        - Higher Test R²
        - Reasonable difference between train and test (avoiding overfitting)

* **4.Results Recording**
    - All the results are stored in a DataFrame called df_results.
    - The results are ordered by the order of the evaluated models (so that the graphical comparison is clear).

* **4.Comparative Visualization**
    - Comparison charts but focused on:
        - RMSE (Entrenamiento vs Prueba).
        - R² (Train vs Test).
    - The Best Model stands out in the charts.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>
1.Load of Results 
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/1.LoadResults.png" width="4000"/>

2.Analysis of Overfitting / Underfitting 
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/2.AnalysisOverfitting.png" width="4000"/>

3.Graphical Comparison
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/3.RMSEvsR.png" width="4000"/>

>[!TIP]
>Don't worry too much yet, because you haven't done Feature Engineering, Optimization, or Data Augmentation.  These steps should improve the results.

>[!NOTE]
>Remember that the workflow is first to do up to this point and then go back to compare after applying the methods to select the best model. 

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**Pandas** for data handling <br> **matplotlib.pyplot** for graphics, and numpy for numerical calculations.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E1.ImportLibraries.png" width="4000"/>|
Load the file **`df_results.csv`**, which contains the evaluation metrics of the models. <br> Print the **`df_results`** table to visualize the results.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E2.LoadResults.png" width="4000"/>|
Define the function **`analyze_overfitting(row)`**, which evaluates whether a model suffers from **overfitting** or **underfitting** using the values of **`Train R²`** and **`Test R²`**. <br> Apply this function to each row of **`df_results`**, adding a new column **`Observation`** with one of these labels: <br> **⚠️ Overfitting** → If **`Train R² > 0.90`** and the difference between **`Train R²`** and **`Test R²`** is greater than **0.15**. <br> **⚠️ Underfitting** → If **`Train R² < 0.5`** and **`Test R² < 0.5`.** <br> **✅ Good Balance** → If the model does not fall into the two previous categories.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E3.AnalysisOverfitting.png" width="4000"/>|
Print the **`df_results`** table after adding the **`Observation`** column. <bt> Order the models based on **`Test RMSE`** (the root mean square error on test data). <br> Select the model with the lowest **`Test RMSE`** as the best model.<br> **`RMSE Test`** is used as the main criterion because a lower error in the test indicates better generalization. <br> **`Test R²`** could be used as an alternative criterion.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E4.SelectionBestModel.png" width="4000"/>|
Set the **`matplotlib`** figure size to (**14**,**6**). <br> Create an array **`x`** with indices of the models for the graphs. <br> Define **width=0.35**, the width of the bars in the chart.<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E5.ComparativeVisualization.png" width="4000"/>|
Create the first bar chart comparing Train RMSE and Test RMSE. <br> Use plt.bar() to represent two sets of bars: <be> Train RMSE → Light blue <br> Test RMSE → Dark blue <br> Adjust the labels and titles of the chart.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E6.RMSEGraph.png" width="4000"/>|

| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E7.HighlightBestModel.png" width="4000"/>|

| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/" width="4000"/>|

| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E8.RGraph.png" width="4000"/>|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E9.Highlight.png" width="4000"/>|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.4EvaluationMetrics/Images/E10.Summary.png" width="4000"/>|