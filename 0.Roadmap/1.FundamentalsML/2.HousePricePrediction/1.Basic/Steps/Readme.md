<h1 align="center"  style="margin-bottom: -10px;">🏠 House Price Prediction with Machine Learning 🏠</h1>
<div align="center">

🌐 This README is available in: [English](Readme.md) | [Spanish](ReadmeESP.md) 🌐

</div>

<h2 align="center">📑 Table of Contents</h2>

1. [Description](#descripcion)
2. [Development Process](#desarollo)
   - [1.1 Data Processing](#datos)
      - [1.1.1 Data Loading](#Cdatos)
      - [1.1.2 Check for Null Values](#Nvalues)
      - [1.1.3 Duplicate Data Detection and Data Type Analysis](#Dduplicados)
   - [1.2 Exploratory Data Analysis (EDA)](#eda)
      - [1.2.1 Heatmap](#heatmap)
      - [1.2.2 Pairplot](#pairplot)
      - [1.2.3 Descriptive Statistics](#desc)
      - [1.2.4 Histograms](#histo)
      - [1.2.5 Boxplot](#boxplot)
        - [1.2.5.1 Boxplot SalePrice](#saleprice)
        - [1.2.5.2 Boxplot Gr Liv Area](#grlivarea)
        - [1.2.5.3 Boxplot OverallQual](#overallqual)
      - [1.2.6 Data Distribution](#distribucion)
      - [1.2.7 Apply Logarithms](#logaritmos)
      - [1.2.8 Normalize the Data](#normalizamos)
    - [1.3 Training Multiple Algorithms](#Malgoritmos)
      - [1.3.1 KNN Regressor](#knn)
      - [1.3.2 SVR (Support Vector Regressor)](#svr)
      - [1.3.3 Neural Networks (MLP)](#mlp)
      - [1.3.4 LightGBM](#gbm)
      - [1.3.5 Ridge Regression (L2 Regularization)](#l2)
      - [1.3.6 Lasso Regression (L1 Regularization)](#l1)
      - [1.3.7 XGBoost](#xgboost)
    - [1.4 Evaluation Metrics](#metricas)
    - [1.5 Optimization (Tuning and Hyperparameters)](#optimizacion)
3. [Group (1/2)](#agrupar1)
4. [Final Results](#resultados)
5. [Technologies Used](#tech)
6. [How to Run](#ejecutar)

<h2 id="descripcion" align="center">📜 Description 📜</h2>

This is a basic regression project implementing a full Machine Learning pipeline (excluding the data augmentation step) to predict house prices. The model considers key features such as quality, size, and location, using historical real estate market data.

Each step is explained below to show the logic used to solve the problem.

<h2 id="desarollo" align="center">Development Process</h2>

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td style="width: 50%; vertical-align: top; text-align: left; padding-right: 20px;">
      The development process of this project follows a logical step-by-step structure to build a robust and interpretable regression model. <br>
      We begin by loading and cleaning the data, followed by an exploratory analysis to better understand the variables and their impact on price. <br>
      Then, multiple algorithms are trained to compare their performance using appropriate metrics. <br>
      For evaluation, we use these metrics to compare and select the best-performing model. <br>
      Afterwards, we optimize the selected model to achieve better results and predictions. <br>
      We then integrate the entire workflow into a single robust program. <br>
      Finally, predictions are made using the trained model and results are displayed in an interactive dashboard.
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/spanish.png?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />
    </td>
  </tr>
</table>

<h3 id="desarollo" align="center">1.1 Data Processing</h3>

<h4 id="Cdatos" align="center">1.1.1 Data Loading</h4>

In the first step, we loaded the Ames Housing dataset, obtaining a total of 2,930 rows and 82 columns.  
This provides a rich and detailed base of features describing the properties, including aspects such as size, quality, location, and more.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/1.dataProcessing/1.1.1.jpeg?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />

> [!NOTE]  
> This stage is crucial to get a general overview of the dataset, identify potential errors or missing data, and plan the next cleaning and analysis steps.

<h4 id="Nvalues" align="center">1.1.2 Check for Null Values</h4>

We checked for null values present in the dataset. At this stage, we only performed a visual inspection to understand the structure of our database, without making decisions yet on which variables to remove or keep.  
This helps us better understand the quality and completeness of the data before proceeding with analysis.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/1.dataProcessing/1.2.jpeg?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />

<h4 id="Dduplicados" align="center">1.1.3 Duplicate Data Detection and Data Type Analysis</h4>

In this step, we combined two important tasks: first, we detected duplicate data to ensure dataset quality; second, we analyzed the data types.  
This is essential for planning future processes, as each variable may require different treatment based on its type.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/1.dataProcessing/1.3.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;" />

<h3 id="eda" align="center">1.2 Exploratory Data Analysis (EDA)</h3>

<h4 id="heatmap" align="center">1.2.1 Heatmap</h4>

First, we created a heatmap of the entire dataset to visualize which variables show relationships with each other. However, due to the large amount of data, the chart does not allow for clear visualization. Therefore, in the next step, we filtered and focused the analysis on the most relevant variables for better interpretation.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.1.1.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">

Next, we printed the most relevant variables to generate a reduced heatmap. The goal is to get a better view and understanding of the relationships between the most influential variables in this specific case.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.1.2.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">

Finally, we used the most relevant variables to generate a reduced heatmap, where we observed that the first two variables show the highest correlation (darker color intensity). This is useful because these variables may be modified or used to improve the model in the next steps.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.1.3.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">

<h4 id="pairplot" align="center">1.2.2 Pairplot</h4>

In this step, a pairplot is used to visualize the behavior of the selected variables. Since we have identified two variables with high correlation, it is important to observe their distribution and visual relationship to better understand how they interact.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.1.4.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">

<h4 id="desc" align="center">1.2.3 Descriptive Statistics</h4>

Based on the selected variables (Gr Liv Area and Overall Qual), we obtain more detailed information about their relationships with the rest of the dataset. However, it is important to remember that SalePrice is our target variable (y), as our main goal is to predict its behavior.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.1.5.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">

<h4 id="histo" align="center">1.2.4 Histograms</h4>

This step helps us visualize the distribution of each variable's data. Through histograms, we can:

- Detect the shape of the distribution (normal, skewed, etc.).
- Identify possible biases and outliers.
- Assess whether a variable requires transformation (such as log or scaling).
- Observe the general distribution across numerical variables.
- Make decisions about preprocessing that could improve model performance.

This visualization is key to better understanding our data before building predictive models.

|Distribution of Gr Liv Area|Distribution of SalePrice|Distribution of Overall Quality|
|------------------------|------------------------|-------------------| 
|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.2.1.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.2.2.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.2.3.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;"> |

>[!IMPORTANT]
>In the next step, we will observe the behavior and combinations of each variable if selected, but we haven’t made any selections yet.

|SalePrice and Gr Liv Area|SalePrice and Overall Qual|SalePrice, Gr Liv Area, Overall Qual|
|------------------------|------------------------|-------------------| 
|Both variables show skewed distributions and contain extreme outliers. Therefore, it is recommended to use MinMaxScaler, as it is safer to preserve the original scale without being affected by these outliers. <br> Alternatively, StandardScaler can also be used, but keep in mind that outliers can significantly influence the mean and standard deviation, affecting the scaling.| SalePrice is a continuous variable, while Overall Qual is an ordinal variable with values from 1 to 10 representing quality. <br> Normalizing Overall Qual is not necessary, as it is a discrete number with specific meaning. Therefore, scaling could be applied only to SalePrice using MinMaxScaler or StandardScaler.| MinMaxScaler is a safe option if we want all variables in the [0, 1] range. <br> StandardScaler may be useful if Gr Liv Area and SalePrice follow a normal distribution. <br> However, Overall Qual is an ordinal variable, so it is advisable to leave it unscaled to preserve its meaning.|

<h4 id="boxplot" align="center">1.2.5 Boxplot</h4>

A boxplot is a graphical representation that shows the distribution of a numerical variable, highlighting its median, quartiles, and possible outliers.<br>
It allows us to quickly and visually identify dispersion, symmetry, and the presence of extreme values.<br><br>

<h5 id="saleprice" align="center">1.2.5.1 Boxplot SalePrice</h5>

We observed several points above the interquartile range, indicating the presence of high outliers. This suggests that there are homes with significantly higher prices than the dataset average.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.3.1.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;"><br>

<h5 id="grlivarea" align="center">1.2.5.2 Boxplot Gr Liv Area</h5>

The Gr Liv Area boxplot also shows the presence of high outliers, with the upper whiskers extending further than in other cases, indicating that some homes have significantly larger living areas than the typical range.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.3.2.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;"><br>

<h5 id="overallqual" align="center">1.2.5.3 Boxplot OverallQual</h5>

The OverallQual boxplot shows values mainly concentrated below the median, with few or no visible outliers, indicating that most homes have a general quality within a limited range.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.3.3.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;"> <br>

Finally, we analyzed the p-value to assess the statistical significance of our variables in relation to the target variable (SalePrice).

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.3.4.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">

<h4 id="distribucion" align="center">1.2.6 Data Distribution</h4>

We need to calculate the skewness of each variable to quantify how skewed its distribution is, which is important for deciding what preprocessing actions to take.
After calculating skewness, we observed that Gr Liv Area and Overall Qual have positive skewness, meaning most values are low, but a few high values skew the distribution.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.3.5.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;"> 

Next, we filtered rows where any variable had a value less than or equal to 0, as in real-world data such values should not exist (e.g., negative area or price makes no sense). This check helps detect errors or inconsistencies in the data before training the model.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/?raw=true" alt="Data types" style="width: 100%; height: auto;"> 

Finally, and importantly, we rechecked for null values, but this time only in the variables we actually care about for the model. This helps us focus preprocessing efforts on relevant columns and make informed decisions about handling missing data.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.3.7.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">

<h4 id="logaritmos" align="center">1.2.7 Log Transformation</h4>

As we observed, the data presents skewed distributions, which can negatively affect the performance of regression models.  
To address this issue, we applied a logarithmic transformation, which helps reduce skewness and makes the distribution more symmetrical.  
Below is a comparison before and after applying the logarithm:

| Before Log Transformation | After Log Transformation |
|---------------------------|--------------------------|
|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.4.1.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.4.2.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">|

<h4 id="normalizamos" align="center">1.2.8 Data Normalization</h4>

Before concluding the distribution analysis, it’s important to consider that variability in feature scales can also affect model performance.  
Therefore, it is necessary to normalize the data so that variables fall within similar ranges. This helps the model learn more efficiently and fairly.  
In this step, we observe the values of selected variables to decide what type of scaling to apply.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.4.1.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">

Finally, we created a graph to verify whether, after normalization and correcting skewness, the data distribution was adequately adjusted.  
This visualization allows us to confirm whether the variable approximates a normal distribution (bell-shaped curve), which is desirable for many machine learning algorithms.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.EDA/2.4.2.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;"> 

<h3 id="Malgoritmos" align="center">1.3 Training Multiple Algorithms</h3>

We now move to the model training stage, where we apply different regression algorithms to compare their performance.  
Since this is a regression problem, the following models are trained:

- KNN Regressor  
- SVR (Support Vector Regressor)  
- Neural Networks (MLP Regressor)  
- Ridge Regression (L2 Regularization)  
- Lasso Regression (L1 Regularization)  
- LightGBM  
- XGBoost  

The goal of this step is to evaluate which model best fits the data based on performance metrics, interpretability, and computational complexity.

<h4 id="knn" align="center">1.3.1 KNN Regressor</h4>

KNN Regressor predicts the value of a point by averaging the values of the k nearest neighbors based on a distance metric.  
It is an instance-based model, with no actual training phase, and works best when similar data points are expected to have similar output values.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.1KNNRegressor.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">

<h4 id="svr" align="center">1.3.2 SVR (Support Vector Regressor)</h4>

SVR is an extension of the Support Vector Machine algorithm for regression tasks.  
It aims to fit a line (or hyperplane) that predicts data within a defined margin of tolerance, minimizing errors outside that margin.  
It is useful for nonlinear data and provides good generalization.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.2SVR.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;"> 

<h4 id="mlp" align="center">1.3.3 Neural Networks (MLP)</h4>

The MLP Regressor (Multi-Layer Perceptron) is a neural network with one or more hidden layers that learns complex patterns through forward and backpropagation.  
It is ideal for capturing nonlinear relationships among variables and adapts well to datasets with multiple features.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.3MLPRegressor.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">

<h4 id="gbm" align="center">1.3.4 LightGBM</h4>

LightGBM is a Gradient Boosting algorithm optimized for speed and efficiency.  
It builds decision trees leaf-wise rather than level-wise, improving performance and accuracy.  
It is ideal for large datasets and regression tasks with high dimensionality.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.4LightGBM.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">  
<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.4LightGBM2.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">

<h4 id="l2" align="center">1.3.5 Ridge Regression (L2 Regularization)</h4>

Ridge Regression is an extension of linear regression that applies L2 regularization to reduce overfitting.  
It penalizes large coefficients by adding their squared sum to the loss function, stabilizing the model when multicollinearity or many variables are present.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.5RidgeRegression.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;"> 

<h4 id="l1" align="center">1.3.6 Lasso Regression (L1 Regularization)</h4>

Lasso Regression uses L1 regularization, which penalizes the absolute sum of the coefficients.  
This not only reduces overfitting but can also eliminate irrelevant variables, as it tends to shrink some coefficients exactly to zero—acting as a form of feature selection.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.6LassoREgressor.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;"> 

<h4 id="xgboost" align="center">1.3.7 XGBoost</h4>

XGBoost (Extreme Gradient Boosting) is a highly optimized and efficient gradient boosting algorithm.  
It uses advanced techniques such as regularization, tree pruning, and parallelization to improve both accuracy and computational performance.  
It is widely popular in machine learning competitions due to its ability to handle complex and noisy data.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.7XGBoost.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;"> 

<h3 id="metricas" align="center">1.4 Evaluation Metrics</h3>

After training multiple regression models, it is essential to evaluate their performance using specific metrics.  
These metrics allow us to objectively compare results and choose the model that best fits the data and generalizes well.  
In this project, the following evaluation metrics were used:

- **MAE (Mean Absolute Error)**: Average of the absolute values of the errors. Measures how far predictions are from actual values on average.  
- **MSE (Mean Squared Error)**: Average of squared errors. Penalizes larger errors more.  
- **RMSE (Root Mean Squared Error)**: Square root of the MSE. Interpreted on the same scale as the target variable.  
- **R² Score (Coefficient of Determination)**: Indicates the percentage of variance in the dependent variable explained by the model. Values close to 1 indicate a good fit.

These metrics help us understand not only how wrong the models are but also the nature of their errors.

After comparing the performance of all models using the evaluation metrics mentioned, **XGBoost** was the algorithm that achieved the best results in terms of accuracy and generalization capability.  
Thanks to its robustness, efficient handling of complex data, and built-in regularization, it is considered the best option to solve this house price prediction problem.

<br>
<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/4.EvaluationMetrics/4.1Evaluation.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">

<h3 id="optimizacion" align="center">1.5 Optimization (Tuning & Hyperparameters)</h3>

Hyperparameter optimization involves manually or automatically adjusting parameters that are not learned during training, such as tree depth, learning rate, or the number of estimators.  
These values have a direct impact on the model’s performance.

|Random Search|Optuna|Early Stopping|
|----------------------------------------------|--------------|----------|
|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/5.Optimization/5.1RandomS.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/5.Optimization/5.2Optuna.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/5.Optimization/5.3EarlyStopping.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;"> 

Finally, we compare the base model with optimized versions obtained through hyperparameter tuning, aiming to find the best possible combination.  
This comparison allows us to visualize whether the optimization truly improves the model’s performance and to choose the final configuration that offers the best results in terms of accuracy and generalization.

|Base GBM|LightGBM + Optuna|LightGBM + Early Stopping|
|----------------------------------------------|--------------|----------|
|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/5.Optimization/5.1RandomS.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/5.Optimization/5.4.2GBMOptuna.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/5.Optimization/5.4.3GBMEarly.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;"> 

<h2 id="agrupar1" align="center">3. Grouping (1/2)</h2>

<h2 id="resultados" align="center">4. Final Results</h2>

📂 **Dataset used:** [AmesHousing.csv](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset) *(available on Kaggle)*  
🧠 **Learning type:** Supervised  
📈 **Problem type:** Regression  
⚙️ **Main algorithm:** LightGBM  
🧪 **Model level:** Basic  
💻 **Language used:** Python  
👤 **Project type:** Personal / Portfolio

<h2 id="tech" align="center">5. Technologies Used</h2>

📊 **Data manipulation and analysis**  
- Pandas  
- NumPy  
- SciPy

📈 **Visualization**  
- Matplotlib  
- Seaborn  
- Altair

🤖 **Machine Learning**  
- Scikit-learn  
- LightGBM  
- Optuna *(for hyperparameter tuning)*

<h2 id="ejecutar" align="center">6. How to Run the Program</h2>

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/6.Agrupar(1-2)/6.1.jpeg?raw=true" alt="Data Types" style="width: 100%; height: auto;">

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Local
python 1.LOGICA.py
```

>[!IMPORTANT]
>These commands are exclusively used to run the program logic, including the complete Machine Learning workflow.