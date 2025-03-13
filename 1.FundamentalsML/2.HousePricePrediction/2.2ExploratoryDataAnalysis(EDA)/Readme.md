<p align = "center" >
    <h1 align = "Center"> 2.1 Data Processing</h1>
</p>

This script performs an exploratory data analysis (EDA) on the AmesHousing_cleaned.csv dataset. It uses Python libraries such as Pandas, NumPy, Matplotlib, Seaborn to analyze and visualize the distribution of house prices and their relationship with other variables.

Various graphical representations are generated, such as histograms, scatter plots, boxplots, and a correlation matrix, which help to better understand the trends and relationships between the variables in the dataset.

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

This analysis helps to detect trends, outliers, and patterns in the data, facilitating decision-making for feature engineering and model selection. 

- Obtain descriptive statistics from the dataset.
- Visualize the distribution of the sale price (**`saleprice`**).
- Analyze relationships between key variables, such as living area (**`gr_liv_area`**) and the sale price.
- Explore the correlation between numerical variables using a correlation matrix.
- Transform the prices with logarithms to improve the normality in the data.
- Evaluate categorical data using box plots and category counts.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

|**Description**|**Histogram of sale Price**|
|----------|---------------------|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/1.HistogramSalePrice.png" width="4000"/>|
|**Description**| **Box of Sale Price**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.BoxplotSalePrice.png" width="4000"/>|
|**Description**|**Living Area vs Sale Price**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/3.LivingAreaSalePrice.png" width="4000"/>|
|**Description**|**Correlation Matrix**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/4.CorrelationMatrix.png" width="4000"/>|
|**Description**|**Histogram of log sale Price**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/5.HistogramLogSalePrice.png" width="4000"/>|
|**Description**|**Living Area vs log Sale Price**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/6.LivingAreaLogSalePrice.png" width="4000"/>|
|**Description**|**Sale price by house style**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/7.SalePriceHouseStyle.png" width="4000"/>|
|**Description**|**Sale Price by Neighborhood**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/8.SalePriceByNeighborhood.png" width="4000"/>|
|**Description**|**Count of Overall Quality**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/9.CountOverallQuality.png" width="4000"/>|
|**Description**|**Name of Colums**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/10.Columns.png" width="4000"/>|
|**Description**|**Estadistic summary**|
|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/11.StadisticSummary.png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.1ImportLibraries.png" width="4000"/>|

add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.2LoadDataSet.png" width="4000"/>|

add|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.4StatisticalSummary.png" width="4000"/>|


add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.5HistogramSalePrice.png" width="4000"/>|


add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.6BoxPlotScalePrice.png" width="4000"/>|

add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.7ScatterPlot.png" width="4000"/>|


add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.8CorrelationMatrix.png" width="4000"/>|

add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.9LogarithmicTransformation.png" width="4000"/>|

add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.10ScatterPlot.png" width="4000"/>|


add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.11AnalysisCategorical.png" width="4000"/>|