<p align = "center" >
    <h1 align = "Center"> Exploratory Data Analysis (EDA) </h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

Explore, visualize, and transform key variables to better understand the dataset and lay the groundwork for predictive modeling.

Visually and statistically explore the already cleaned data (AmesHousing_cleaned.csv), focusing on understanding the distribution of SalePrice, looking for key relationships between variables, and paving the way for modeling (for example, discovering the need to apply transformations like the logarithm).

* **1.Data Load** 
    - The clean dataset you generated in Program 1 (1.1DataProcessing) (AmesHousing_cleaned.csv), now without null values, is loaded.

* **2.Visual Analysis: Histograms (SalePrice)**
    - Show how house prices are distributed.
    - Objective: To see if the distribution is normal, skewed, or has outliers.
    - Result: Generally shows that prices are right-skewed (lower prices are more frequent, but there are some very high prices).

* **3.Visual Analysis (Boxplot SalePrice)**
    - Show the price dispersion and detect possible outliers.
    - Objective: To see interquartile ranges and detect outliers.
    - Result: In this type of datasets, it is common to see some extremely high prices as outliers.

* **4.Relationship between Variables (Scatter Plot Gr Liv Area vs Sale Price)**
    - Show how the livable area (Gr Liv Area) relates to the sale price.
    - Objective: To see if there is a linear relationship or interesting patterns.
    - Result: Normally, a positive trend is observed (the larger the area, the higher the price), but with some outliers (very large houses with lower-than-expected prices).

* **5. Correlation Matrix (Numerical Variables)**
    - Visualize the correlation between all numerical variables.
    - Objective: Identify which variables have a high correlation with SalePrice, which helps in selecting features for the model.
    - Result: In general, variables such as OverallQual, GrLivArea, and GarageCars tend to have a high correlation with SalePrice.

* **6. Logarithmic Transformation of SalePrice (Log Transform)**
    - A new column is created
    - Objective: If SalePrice is skewed (as it often is), applying the logarithm helps to normalize the distribution.
    - Result: A new histogram shows that Log_SalePrice is more symmetrical.

* **7. Scatter Plot with Log_SalePrice (Gr Liv Area vs Log SalePrice)**
    - Understand your data before training any modelRepite el scatter plot pero con Log_SalePrice.
    - Objective: To see if the linear relationship improves by using the logarithm.
    - Result: Normally, a more linear relationship is seen, which is better for linear models.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

|SalePrice Histogram|Boxplot Sale Price|
|----------|---------------------|
|See Initial Distribution <br> <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/Images/1.HistogramSalePrice.png" width="4000"/>|Detect Outliers <br> <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/Images/2.BoxplotSalePrice.png" width="4000"/>|
|<p align = "center" >**GrLivArea vs SalePrice Scatter Plot**</p>|<p align = "center" >**Correlation Matrix**</p>|
|Direct relationship between area and price <br> <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/Images/3.LivingAreavsSalePRice.png" width="4000"/> |Identify key variables <br> <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/Images/4.CorrelationMatrix.png" width="4000"/>|
|<p align = "center" >**Transformación Logarítmica**</p>|<p align = "center" >**Scatter Plot GrLivArea vs Log_SalePrice**</p>|
|Normalizar SalePrice <br> <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/Images/5.HistogramLogSalePrice.png" width="4000"/> |Mejorar relación lineal <br> <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.2ExploratoryDataAnalysis(EDA)/Images/6.LivingAreaLogSalePrice.png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (PENDING) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** Used to work with arrays and matrices, and it has functions for mathematics and statistics <br> **Pandas:** Used to manipulate and analyze data, particularly with DataFrames (tables of rows and columns).<br> **matplotlib.pyplot:** Used to create graphs, and pyplot is specifically for generating different types of plots (e.g., bar, line, scatter).<br> **load_digits from sklearn.datasets:** Used to load a dataset of handwritten digits, which is commonly used in classification projects.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/1.DeclareteLibraries.png" width="4000"/>|
**`digits`:** loads a dataset of handwritten digit images, where you find images of numbers 0 to 9 in black and white, each with 8x8 pixels. Each image represents a number (digit) and has a label indicating which number it is. <br> **`x`:** Contains images (numeric format), for each 8x8 image it is flattened into a 64 array, each value represents the pixel intensity (0 = black, 16 = white). <br> **`y`:** contains the labels (real numbers of the images), each element is a number from 0 to , representing which digit each image is| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/2.LoadDataset.png" width="4000"/>|
**`x.shape`:** Returns the dimension of x(1797, 64), it has 1797 images, and each image has 64 values (8x8 pixels). <br> **`y.shape`:** Returns the dimension of y (1797), there are 1797 labels per image, each label is the number it represents (0-9).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/3.ExploringDimensionsDataSet.png" width="4000"/>|
**`clases`:** Contains the unique values of y(0-9) <br> **`count_classes`:** Array that indicates how many examples there are of each class | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/4.CheckClassesNumberExamples.png" width="4000"/>|
**`plt.figure(figsize=(8,5))`:** Set the size of the figure (8 wide and 5 high) <br> **`plt.bar(clases, count_classes, color='skyblue')`:** Create the bar chart, the list that contains the number of examples, assign the color of the bars. <br> **`plt.xlabel('Digit')`:** Set the x-axis label <br> **`plt.ylabel('Number of examples')`:** Set the y-axis label <br> **`plt.title('Distribution of the classes')`:** Add a title to the chart. <br> **`plt.show()`:** Visualize the graph | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/5.ViewDistributionClasses.png" width="4000"/>|
**`fig, axes = plt.subplots(2, 5, figsize=(10, 5))`:** Create a figure and a grid (2 rows and 5 columns) totaling 10 subplots, set the size (10, 5 in inches)  10 wide and 5 high <br> **`fig.suptitle("Examples of images")`:** Establish a general title <br> **`for i, ax in enumerate(axes.ravel())`:** Iterates through each of the subplots, converts the (2x5) matrix into a one-dimensional array, which makes individual access easier, and returns both the index `i` and the `ax` object in each iteration. <br> **`ax.imshow(x[i].reshape(8,8), cmap='gray')`:** it is a vector image from the dataset.  resize to an 8x8 matrix, apply a grayscale <br> **`ax.set_title(f"label: {y[i]}")`:** Assign a title to each subplot. <br> **`ax.axis('off')`:** Deactivate the axes so that the marks or values do not appear. <br> **`plt.show()`:** Show the subplots and images. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/6.ViewExample.png" width="4000"/>|