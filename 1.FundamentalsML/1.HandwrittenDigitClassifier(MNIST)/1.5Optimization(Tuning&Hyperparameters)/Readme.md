<p align = "center" >
    <h1 align = "Center"> Optimization Tuning and D </h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

Optimize the models through hyperparameter tuning and compare their performance in the classification of MNIST 8x8 digits.

* **1.Load and prepare the data** 
    - Load the MNIST 8x8 dataset.
    - Split into training and test data.
    - Normalize the data with **`StandardScaler`**.

* **2.Optimization with GridSearchCV (Tuning)**
    - Different hyperparameters are tested for 4 models:
        - Logistic Regression
        - K-Nearest Neighbors (KNN)
        - Support Vector Machines (SVM)
        - MLP neural network
    - **`GridSearchCV`** is Used to find the best hyperparameters.

* **3.Evaluate Performance**
    - The best model found is saved.
    - Its accuracy is measured on the test set.
    - The confusion matrix is generated to analyze errors.
    - The classification report is shown with metrics such as precision and recall.

* **4.Visualization of results**
    - The comparison of accuracy between models is graphed.
    - Confusion matrices with heatmaps are shown.

<p align = "center" >
    <h2 align = "Center">📝 Results (PENDING Before after)📝 </h2>
</p>

<p align = "center" >
    <h3 align = "Center">1.Hyperparameter</h3>
</p>

After optimizing the hyperparameters of each model using GridSearchCV, they were evaluated on the test set.
 **Acurracy Without Hyperparameter**| **Acurracy with Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/Data.png" width="800"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/Data.png" width="1000"/>|

<p align = "center" >
    <h3 align = "Center">2.Bars Graph</h3>
</p>
Show the improved accuracy of each model so you can compare them visually.

<p align = "center" >
    <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/Results.png" width="500"/>
</p>
<p align = "center" >
    <h3 align = "Center">3.Confusion Matrix</h3>
</p>
It's a table that shows how many times the model correctly classified each number and how many times it made mistakes.
**To read it**
- The rows represent the real numbers (true labels).
- The columns represent the model's predictions.
- The values on the main diagonal are the correct predictions (true positives).
- The values outside the diagonal are errors (incorrect predictions).

| **Logistic Regression**| **K-Nearest Neighbors(KNN)**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ConfusionMatrix(LR).png" width="4000"/> |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ConfusionMatrix(KNN).png" width="2000"/>|
| <p align = "center" > **SVM** </p> | <p align = "center" > **Multi Layer Perceptron(MLP)** </p>|
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ConfusionMatrix(SVM).png" width="4000"/> |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ConfusionMatrix(MLP).png" width="2000"/>|


4.Classification Report 


| **Logistic Regression**| **K-Nearest Neighbors(KNN)**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ReportClassification(LR).png" width="4000"/> |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ReportClassification(KNN).png" width="2000"/>|
| <p align = "center" > **SVM** </p> | <p align = "center" > **Multi Layer Perceptron(MLP)** </p>|
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ReportClassification(SVM).png" width="4000"/> |<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ReportClassification(MLP).png" width="2000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻 (PENDING)</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** Used to work with arrays and matrices, and it has functions for mathematics and statistics <br> **Pandas:** Used to manipulate and analyze data, particularly with DataFrames (tables of rows and columns).<br> **matplotlib.pyplot:** Used to create graphs, and pyplot is specifically for generating different types of plots (e.g., bar, line, scatter).<br> **load_digits from sklearn.datasets:** Used to load a dataset of handwritten digits, which is commonly used in classification projects.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/1.DeclareteLibraries.png" width="4000"/>|
**`digits`:** loads a dataset of handwritten digit images, where you find images of numbers 0 to 9 in black and white, each with 8x8 pixels. Each image represents a number (digit) and has a label indicating which number it is. <br> **`x`:** Contains images (numeric format), for each 8x8 image it is flattened into a 64 array, each value represents the pixel intensity (0 = black, 16 = white). <br> **`y`:** contains the labels (real numbers of the images), each element is a number from 0 to , representing which digit each image is| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/2.LoadDataset.png" width="4000"/>|
**`x.shape`:** Returns the dimension of x(1797, 64), it has 1797 images, and each image has 64 values (8x8 pixels). <br> **`y.shape`:** Returns the dimension of y (1797), there are 1797 labels per image, each label is the number it represents (0-9).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/3.ExploringDimensionsDataSet.png" width="4000"/>|
**`clases`:** Contains the unique values of y(0-9) <br> **`count_classes`:** Array that indicates how many examples there are of each class | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/4.CheckClassesNumberExamples.png" width="4000"/>|
**`plt.figure(figsize=(8,5))`:** Set the size of the figure (8 wide and 5 high) <br> **`plt.bar(clases, count_classes, color='skyblue')`:** Create the bar chart, the list that contains the number of examples, assign the color of the bars. <br> **`plt.xlabel('Digit')`:** Set the x-axis label <br> **`plt.ylabel('Number of examples')`:** Set the y-axis label <br> **`plt.title('Distribution of the classes')`:** Add a title to the chart. <br> **`plt.show()`:** Visualize the graph | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/5.ViewDistributionClasses.png" width="4000"/>|
**`fig, axes = plt.subplots(2, 5, figsize=(10, 5))`:** Create a figure and a grid (2 rows and 5 columns) totaling 10 subplots, set the size (10, 5 in inches)  10 wide and 5 high <br> **`fig.suptitle("Examples of images")`:** Establish a general title <br> **`for i, ax in enumerate(axes.ravel())`:** Iterates through each of the subplots, converts the (2x5) matrix into a one-dimensional array, which makes individual access easier, and returns both the index `i` and the `ax` object in each iteration. <br> **`ax.imshow(x[i].reshape(8,8), cmap='gray')`:** it is a vector image from the dataset.  resize to an 8x8 matrix, apply a grayscale <br> **`ax.set_title(f"label: {y[i]}")`:** Assign a title to each subplot. <br> **`ax.axis('off')`:** Deactivate the axes so that the marks or values do not appear. <br> **`plt.show()`:** Show the subplots and images. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/6.ViewExample.png" width="4000"/>|