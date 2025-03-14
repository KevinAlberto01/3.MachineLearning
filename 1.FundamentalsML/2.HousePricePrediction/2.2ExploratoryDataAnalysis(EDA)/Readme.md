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

<p align = "center" >
    <h3 align = "Center"> Histogram of Sale Price </h3>
</p>

**Positive skew:** Since the distribution is not symmetrical and has a long tail to the right, this can affect some regression models. <br> It may be useful to apply a logarithmic transformation to the prices to make the distribution more normal. <br> **Outliers:** There are some houses with very high prices (outliers).  so it is possible to handle them, either by removing them or using them carefully to avoid them having too much influence on the model. <br>**Price ranged:** Most prices are between $100,000 and $300,000, so any prediction outside this range should be evaluated carefully. <br>

|<p align = "left"> **`X-axis (Sale Price)`:** Represents the sale prices of the houses. <br> Y-axis (Frequency): Indicates how many houses fall within a certain price range. <br> Shape of the distribution: Most houses are priced in the range of $100,000 to $200,000, which means the distribution is right-skewed (there are fewer high-value houses). <br> Long tail to the right: There are some houses with very high prices (over $500,000), but they are less common. <br> Mode: The highest peak indicates the most frequent price of houses.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/1.HistogramSalePrice.png" width="4000"/>|
|--------------------------------------------|-----------------------------------------------|

<p align = "center" >
    <h3 align = "Center"> Box of Sale Price </h3>
</p>

**Box (blue rectangle):** Represents the **interquartile range (IQR)**, that is, the range where 50% of the data is found (between the first quartile Q1 and the third quartile Q3). <br> **Line inside the box:** Represents the **median** (central value of the data). <br> **Whiskers (horizontal lines on the sides of the box):** Represent the range of data without outliers.<br> They extend up to 1.5 times the interquartile range. <br> **Points outside the whiskers:** They are **outliers** (extreme values). In this case, there are many high values that are considered outliers.

|<p align = "left"> **Price distribution:** <br> Most houses are priced between approximately **$100,000** and **$250,000.** <br> The median (line in the box) is close to **$180,000**, indicating that half of the houses cost less than this value. <br> **Outliers in house prices:** <br> There are many houses with prices well above the median, some reaching up to **$700,000**. <br> These extreme values can influence Machine Learning models, especially methods like **Linear Regression**, as they can skew the prediction. <br> **Possible solution for outliers:** <br> **Remove outliers:** If they are errors or very rare cases, you could remove them (for example, if they represent less than 1% of the data). <br> **Logarithmic transformation:** Applying log(Sale Price) can reduce the impact of extreme values. <br> **Robust models:** Some algorithms like **Random Forest or Gradient Boosting** handle outliers better than simple linear regression.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.BoxplotSalePrice.png" width="4000"/>|
|--------------------------------------------|-----------------------------------------------|

<p align = "center" >
    <h3 align = "Center"> Living Area vs Sale Price </h3>
</p>

**Positive relationship:** An **increasing trend** is observed, indicating that the larger the living area, the higher the house price.
- This suggests that **`gr_liv_area`** is an important feature for predicting the house price.

**Non-strictly linear pattern:** Although there is a clear relationship, the dispersion of the points suggests that other factors also influence the price.

**Outliers in the data:**
- There are some points that deviate from the general trend.
- For example, there are houses with large areas (>4000 square feet) but with low prices, which is unusual.
- There are also houses with extremely high prices that could influence the model.

|<p align = "left"> **`X-axis (gr_liv_area)`:** Represents the living area in square feet. <br> **`Y Axis (Sale Price)`:** Represents the sale price in dollars. <br> **`Each blue dot:`** Represents a house, with its living area and corresponding sale price.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/3.LivingAreaSalePrice.png" width="4000"/>|
|--------------------------------------------|-----------------------------------------------|

<p align = "center" >
    <h3 align = "Center"> Correlation Matrix (PENDING)</h3>
</p>

|<p align = "left">           |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/4.CorrelationMatrix.png" width="4000"/>|
|--------------------------------------------|-----------------------------------------------|

<p align = "center" >
    <h3 align = "Center"> Histogram of log sale Price (PENDING)</h3>
</p>

|          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/5.HistogramLogSalePrice.png" width="4000"/>|
|--------------------------------------------|-----------------------------------------------|

<p align = "center" >
    <h3 align = "Center"> Living Area vs log Sale Price (PENDING)</h3>
</p>

|<p align = "left">          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/6.LivingAreaLogSalePrice.png" width="4000"/>|

<p align = "center" >
    <h3 align = "Center"> Sale price by house style (PENDING)</h3>
</p>

|<p align = "left">          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/7.SalePriceHouseStyle.png" width="4000"/>|

<p align = "center" >
    <h3 align = "Center"> Sale Price by Neighborhood (PENDING)</h3>
</p>

|<p align = "left">          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/8.SalePriceByNeighborhood.png" width="4000"/>|

<p align = "center" >
    <h3 align = "Center"> Count of Overall Quality (PENDING)</h3>
</p>

|<p align = "left">           |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/9.CountOverallQuality.png" width="4000"/>|

<p align = "center" >
    <h3 align = "Center"> Name of Colums (PENDING)</h3>
</p>

|<p align = "left">           |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/10.Columns.png" width="4000"/>|

<p align = "center" >
    <h3 align = "Center"> Estadistic summary (PENDING)</h3>
</p>

|<p align = "left">          |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/11.StadisticSummary.png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**pandas:** To load and manipulate tabular data (DataFrames). <br> **numpy:** For numerical operations and mathematical functions. <br> **matplotlib.pyplot:** For graphical visualizations. <br> **seaborn:** For more stylish statistical graphics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.1ImportLibraries.png" width="4000"/>|
**`file_path`:** Path where the CSV file with the cleaned data is stored. <br> **`pd.read_csv(file_path)`:** Loads the CSV into a DataFrame (**`df`**).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.2LoadDataSet.png" width="4000"/>|
**`str.strip()`:** Removes spaces in column names (prevents errors when accessing them). <br> **`print(df.columns)`:** Displays the column names in the console.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.3CleaningColumnNames.png" width="4000"/>|
**`df.describe()`:** Generates descriptive statistics (mean, standard deviation, percentiles) for all numeric columns.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.4StatisticalSummary.png" width="4000"/>|
**`plt.hist()`:** Creates a histogram of the sale prices (**`saleprice`**). <br> **`dropna()`:** Removes null values. <br> **`bins=50`:** Divide the data into 50 intervals. <br> **Color:** **`#87CEEB`** (Sky Blue). <br> **`grid(True, linestyle='--', alpha=0.6)`:** Adds a semi-transparent dotted grid.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.5HistogramSalePrice.png" width="4000"/>|
**`plt.boxplot()`:** Draws a horizontal boxplot of the selling price. <br> **`patch_artist=True`:** Allows changing the color of the box. <br> **Box color:** **`#87CEEB`** (Sky Blue).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.6BoxPlotScalePrice.png" width="4000"/>|
**`plt.scatter(x, y)`:** Shows the relationship between gr_liv_area (living area) and saleprice (sale price). <br> **Color of dots: #4682B4 (Steel Blue).<br> alpha=0.5: Makes the points semi-transparent for better visualization.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.7ScatterPlot.png" width="4000"/>|
**`df.corr()`:** Calculates the correlation between numerical variables. <br> **`plt.imshow()`:** Draws the correlation matrix. <br> **`Color Map:`** coolwarm (rojo-azul). <br> **`plt.xticks()`** and **`plt.yticks()`:** Label the rows and columns.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.8CorrelationMatrix.png" width="4000"/>|
**`np.log1p(x)`:** Applies `log(1 + x)` to `saleprice` to reduce the skewness of the distribution. <br> **Color:** **`#00BFFF`** (Deep Sky Blue).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.9LogarithmicTransformation.png" width="4000"/>|
Show the relationship between **`gr_liv_area`** and the price on a logarithmic scale. <br> **Color:** **`#1E90FF`** (Dodger Blue).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.10ScatterPlot.png" width="4000"/>|
Remove rows where **`house_style`** has null values. <br> Boxplot by Housing Style Boxplot showing the distribution of prices for each type of housing (**`house_style`**). <br> Color palette: **`Blues`**. <bt> Boxplot of prices by **`neighborhood`.** <br> General Quality Housing Count
Count how many houses there are by quality level (**`overall_qual`**). | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.2ExploratoryDataAnalysis(EDA)/Images/2.11AnalysisCategorical.png" width="4000"/>|