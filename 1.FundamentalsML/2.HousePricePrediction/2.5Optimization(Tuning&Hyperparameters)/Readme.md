<p align = "center" >
    <h1 align = "Center"> Optimization (Tuning & Hyperparameters)</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

The search for the best hyperparameters for 4 key models (**Linear, Trees, Forests, and KNN**), comparing them and visualizing their metrics to identify the optimal model for predicting SalePrice.
Perform hyperparameter optimization for 4 regression models and evaluate which configuration offers the best balance between accuracy (R²) and error (RMSE) when predicting house prices (SalePrice). The approach is to compare multiple combinations of hyperparameters to improve the performance of each model.

* **1.Data Loading and Preparation** 
    - Load the cleaned dataset (AmesHousing_cleaned.csv).
    - Separate Features (X) and Target (y).
    - Apply get_dummies() to convert categorical variables.
    - Scale the variables with StandardScaler or RobustScaler, as appropriate.
    - Divide the dataset into train and test (80%-20%).

* **2.Models and Methods of Optimization**
    - The program optimizes and compares the following models:
        - LinearRegression
            - Simple linear regression (without hyperparameters, only serves as a baseline).
        - DecisionTreeRegressor
            - Decision tree with depth and criteria optimization.
        - RandomForestRegressor
            - Tree ensemble (Random Forest) with optimization of the number of trees and depth.
        - KNeighborsRegressor
            - K-nearest neighbors (KNN) regressor optimizing the number of neighbors and metrics.

* **3.Evaluated Metrics**
    - For each combination of hyperparameters, the following metrics are calculated:
        - Train RMSE
            - Root mean square error on the training set.
        - Test RMSE
            - Root mean square error on the test set.
        - Train R²
            - Coefficient of determination in the training set.
        - Test R²
            - Coefficient of determination in the test set.
    - These metrics are key to evaluating accuracy and detecting overfitting or underfitting.
        

* **4.Results Record**
    - The results are stored in a DataFrame called df_results.
    - This DataFrame is sorted and saved in a CSV file.
    - It is used to consult and analyze the best hyperparameters found for each model.

* **5.Graphical Comparison**
    - Two comparative charts are generated using matplotlib:
    - Comparison of RMSE
        - Bar chart comparing the error (Train/Test) of each model.
    - Comparison of R²
        - Bar chart comparing the accuracy (Train/Test) of each model.
    - These visualizations allow you to quickly identify which model is more accurate and which suffers from overfitting or underfitting.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center">1.Summary of SalePrice </h4>
</p>

Is the result of **`df['saleprice']. describe()`**, which provides key statistics about the variable saleprice (the sale price of the houses).

|<p align = "left"> **`count` (2930)** →  It is the number of records in the **`saleprice`** column.  That is to say, you have data on 2930 houses. <br> **`mean` (180,796.06)** → It is the average selling price of the houses. <br> **`std` (79,886.69)** → It is the standard deviation, which indicates how much the prices vary from the average. <br> **`min` (12,789)** → It is the lowest recorded selling price. <br> **`25%` (129,500)** → It is the first quartile.  It means that 25% of the houses have a price less than or equal to $129,500. <br> **`50%` (160,000)** → It is the median.  Half of the houses cost less than $160,000 and the other half more. <br> **`75%` (213,500)** → It is the third quartile.  75% of the houses cost less than or equal to $213,500. <br> **`max` (755,000)** → It is the highest recorded selling price.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/Images/SummarySalePrice.png" width="4000"/>|
|----------|---------------------|

The prices are very dispersed (large difference between minimum and maximum).
The median ($160,000) is lower than the average ($180,796), which suggests that there are some houses with very high prices that are skewing the average upwards.
There could be outliers, especially if a few houses have extremely high or low prices.
Distribution of SalePrices

<p align = "center" >
    <h4 align = "Center">2.Distribution of SalePrice (PEnding)</h4>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/Images/DistributionSalePrice.png" width="4000"/>

<p align = "center" >
    <h4 align = "Center">3.Results </h4>
</p>

|<p align = "left">It is recommended to use RobustScaler instead of StandardScaler because there are high values in X, indicating the possible presence of outliers. <br> RobustScaler will help ensure that the scaling is not affected by these extreme values. |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/Images/Results.png" width="4000"/>|
|----------|---------------------|

<p align = "center" >
    <h4 align = "Center">4.log1p(yes) vs log1p(no) </h4>
</p>

If the values of **`SalePrice`** are very dispersed or have a large difference between low and high prices, this transformation helps to: 
* **Reduce the variability** of the data, preventing extreme values from dominating the model. 
* **Make the distribution more normal (symmetric)**, which improves the accuracy of some Machine Learning algorithms.

**When to answer “y” (yes)**
* If house prices have a very skewed distribution to the right (long tail with very high values).
* If the models are not working well because high values are affecting the calculations too much.

**When to answer “n” (no)**
* If SalePrice already has a distribution close to normal.
* If you prefer to work with the original values without modifications.

<p align = "center" >
    <h4 align = "Center">4.1 Comparation </h4>
</p>

|log1p yes|log1p not|
|----------|---------------------|
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/Images/ComparationModels(Yes).png" width="4000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/Images/ComparaqtionModels(not).png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center">Model Interpretation </h4>
</p>

**Linear Regression** 
* Good overall performance with a Test_R2 of 0.845 (84.5% of variance explained).
* The difference between Train_RMSE (18786) and Test_RMSE (35230) suggests that the model may not generalize perfectly.

**Decision Tree**
* Train_RMSE of 0 and Train_R2 of 1.0 → the model memorizes the data perfectly (overfitting).
* Test_RMSE of 35128 indicates that on new data it does not perform as well.
* Test_MAE of 24304 suggests large errors in new predictions.

**Random Forest.**
* Best model overall:
    * Test_R2 of 0.909 (explains 90.9% of variability).
    * Test_RMSE of 26993, the lowest of all models.
    * Test_MAE of 16029, the lowest absolute error after Linear Regression.
    * Seems to strike a good balance between accuracy and generalization.

**KNN (K-Nearest Neighbors).**
* Worst model:
    * Test_R2 of 0.697, the lowest (poor generalization).
    * Test_RMSE of 49282, the highest (highest error).
    * Test_MAE of 28889, the worst in absolute error.
    * KNN does not seem to be a good choice for this problem.

|Comparation of Models yes|Comparation of Models not|
|----------|---------------------|
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/Images/ModelComparationResults(yes).png" width="4000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.5Optimization(Tuning%26Hyperparameters)/Images/ModelComparationResults(not).png" width="4000"/>|


<p align = "center" >
    <h4 align = "Center">4.1 Comparation </h4>
</p>
|RMSEvsR2 yes|RMSEvsR2 not|
|----------|---------------------|
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.5Optimization(Tuning%26Hyperparameters)/Images/RMSEvsR2(yes).png" width="4000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.5Optimization(Tuning%26Hyperparameters)/Images/RMSEvsR2(not).png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (PENDING)💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** Used to work with arrays and matrices, and it has functions for mathematics and statistics <br> **Pandas:** Used to manipulate and analyze data, particularly with DataFrames (tables of rows and columns).<br> **matplotlib.pyplot:** Used to create graphs, and pyplot is specifically for generating different types of plots (e.g., bar, line, scatter).<br> **load_digits from sklearn.datasets:** Used to load a dataset of handwritten digits, which is commonly used in classification projects.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/1.DeclareteLibraries.png" width="4000"/>|
**`digits`:** loads a dataset of handwritten digit images, where you find images of numbers 0 to 9 in black and white, each with 8x8 pixels. Each image represents a number (digit) and has a label indicating which number it is. <br> **`x`:** Contains images (numeric format), for each 8x8 image it is flattened into a 64 array, each value represents the pixel intensity (0 = black, 16 = white). <br> **`y`:** contains the labels (real numbers of the images), each element is a number from 0 to , representing which digit each image is| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/2.LoadDataset.png" width="4000"/>|
**`x.shape`:** Returns the dimension of x(1797, 64), it has 1797 images, and each image has 64 values (8x8 pixels). <br> **`y.shape`:** Returns the dimension of y (1797), there are 1797 labels per image, each label is the number it represents (0-9).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/3.ExploringDimensionsDataSet.png" width="4000"/>|
**`clases`:** Contains the unique values of y(0-9) <br> **`count_classes`:** Array that indicates how many examples there are of each class | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/4.CheckClassesNumberExamples.png" width="4000"/>|
**`plt.figure(figsize=(8,5))`:** Set the size of the figure (8 wide and 5 high) <br> **`plt.bar(clases, count_classes, color='skyblue')`:** Create the bar chart, the list that contains the number of examples, assign the color of the bars. <br> **`plt.xlabel('Digit')`:** Set the x-axis label <br> **`plt.ylabel('Number of examples')`:** Set the y-axis label <br> **`plt.title('Distribution of the classes')`:** Add a title to the chart. <br> **`plt.show()`:** Visualize the graph | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/5.ViewDistributionClasses.png" width="4000"/>|
**`fig, axes = plt.subplots(2, 5, figsize=(10, 5))`:** Create a figure and a grid (2 rows and 5 columns) totaling 10 subplots, set the size (10, 5 in inches)  10 wide and 5 high <br> **`fig.suptitle("Examples of images")`:** Establish a general title <br> **`for i, ax in enumerate(axes.ravel())`:** Iterates through each of the subplots, converts the (2x5) matrix into a one-dimensional array, which makes individual access easier, and returns both the index `i` and the `ax` object in each iteration. <br> **`ax.imshow(x[i].reshape(8,8), cmap='gray')`:** it is a vector image from the dataset.  resize to an 8x8 matrix, apply a grayscale <br> **`ax.set_title(f"label: {y[i]}")`:** Assign a title to each subplot. <br> **`ax.axis('off')`:** Deactivate the axes so that the marks or values do not appear. <br> **`plt.show()`:** Show the subplots and images. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/6.ViewExample.png" width="4000"/>|