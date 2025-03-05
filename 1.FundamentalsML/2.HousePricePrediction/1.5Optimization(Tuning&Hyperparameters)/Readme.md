<p align = "center" >
    <h1 align = "Center"> Optimization (Tuning & Hyperparameters)</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

The search for the best hyperparameters for 4 key models (Linear, Trees, Forests, and KNN), comparing them and visualizing their metrics to identify the optimal model for predicting SalePrice.
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

✅ Carga y limpia los datos.
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.4EvaluationMetrics/Images/1.LoadResults.png" width="4000"/>
✅ Aplica get_dummies().
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.4EvaluationMetrics/Images/1.LoadResults.png" width="4000"/>
✅ Escala las variables.
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.4EvaluationMetrics/Images/1.LoadResults.png" width="4000"/>
✅ Opcionalmente aplica log1p() a SalePrice.
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.4EvaluationMetrics/Images/1.LoadResults.png" width="4000"/>
✅ Entrena los 4 modelos: Linear Regression, Decision Tree, Random Forest y KNN.
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.4EvaluationMetrics/Images/1.LoadResults.png" width="4000"/>
✅ Calcula RMSE y R² de cada modelo (train y test).
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.4EvaluationMetrics/Images/1.LoadResults.png" width="4000"/>
✅ Grafica las comparaciones de RMSE y R² en barras.
<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.4EvaluationMetrics/Images/1.LoadResults.png" width="4000"/>

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