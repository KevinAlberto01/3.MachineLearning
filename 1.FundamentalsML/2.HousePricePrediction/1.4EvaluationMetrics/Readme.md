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
Total loaded shape
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.3TraininigMultipleAlgorithms/Images/7.DatasetLoaded.png" width="4000"/>

Categorical colums 
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.3TraininigMultipleAlgorithms/Images/2.CategoticalColumns.png" width="4000"/>

Best max_depth (Decision Tree Regressor)
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.3TraininigMultipleAlgorithms/Images/8.BestDecisionTreeRegressor.png" width="4000"/>

|Linear Regression|Decision Tree Regressor|
|----------|---------------------|
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.3TraininigMultipleAlgorithms/Images/3.LinealRegression.png" width="4000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.3TraininigMultipleAlgorithms/Images/4.DecisionTreeRegressor.png" width="4000"/>|
|<p align = "center" >**Random Forest Regressor**</p>|<p align = "center" >**KNeighbors Regressor**</p>|
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.3TraininigMultipleAlgorithms/Images/5.RandomForestRegressor.png" width="4000"/> |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.3TraininigMultipleAlgorithms/Images/6.KNeighborsRegressor.png" width="4000"/>|

Comparation RMSE vs R²
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.3TraininigMultipleAlgorithms/Images/1.RMSEvsR2.png" width="4000"/>

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