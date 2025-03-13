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

|Description| Histogram of sale Price|
|----------|---------------------|
|          |                        |
|Description| Box of Sale Price|
|          |                        |
|Description| Living Area vs Sale Price|
|          |                        |
|Description| Correlation Matrix|
|          |                        |
|Description| Histogram of log sale Price|
|          |                        |
|Description| Living Area vs log Sale Price|
|          |                        |
|Description| Sale price by house style|
|          |                        |
|Description| Sale Price by Neighborhood|
|          |                        |
|Description| Count of Overall Quality|
|          |                        |
|Description| Name of Colums|
|          |                        |
|Description| Estadistic summary|
|          |                        |

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.1DataProcessing/Images/1.ImportLibraries.png" width="4000"/>|

add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.1DataProcessing/Images/2.LoadDataset.png" width="4000"/>|

add|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.1DataProcessing/Images/3.DataInspection.png" width="4000"/>|


add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.1DataProcessing/Images/4.ColumnClassification.png" width="4000"/>|


add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.1DataProcessing/Images/5.CleaningData.png" width="4000"/>|

add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.1DataProcessing/Images/6.SaveCleaned.png" width="4000"/>|


add| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.1DataProcessing/Images/7.DataInspection.png" width="4000"/>|