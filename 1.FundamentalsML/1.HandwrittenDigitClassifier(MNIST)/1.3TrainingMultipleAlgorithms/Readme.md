<p align = "center" >
    <h1 align = "Center"> Training Multiple Algorithms</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

Probar diferentes modelos te permite entender cuál funciona mejor para tu problema y cómo varía el rendimiento según el algoritmo.
Este proceso te enseña a no depender de un solo algoritmo y a comparar métodos en la práctica.

* **1.Logistic Regression** 
    - Basic but effective 

* **2.K-Nearest Neighbors(KNN)**
    - Sensitive to Neighbors and data

* **3.Support Vector Machine (SVM)**
    -Tens to be robust for this type of problems 

* **4.Multi-Layer Perceptron(MLP)**
    -Can Perform very well, but it depends on the training

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

That chart visually represents the accuracy of each algorithm you tested. Each bar represents an algorithm, and the height indicates how well it performed.

|<p align = "left"> Higher = better accuracy. <br> Lower = worse accuracy.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/Results.png" width="700"/>|
|----------------------------------------|----------------------------------------------------------------------------|

Percentage of correct answers in the test set (the closer to 1 or 100%, the better).
<p align="center">
  <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/Data.png" alt="Result2" width="1000"/>
</p>

In this exercise we choose the **SVM Algorithm with 97% of Acurrancy** because have better precition and its excelent to small datasets 

<p align = "center" >
    <h2 align = "Center"> 🔎 Aspects to consider 🔍</h2>
</p> 

|Algorithm|Why would you choose it?| When to avoid it? |
|---------|------------------------|-------------------|
|<p align = "center"> svm </p>|<p align = "center"> Better accuracy, works well on small datasets </p>|<p align = "center"> Can be slow on large databasets </p>|
|<p align = "center"> Logistic Regression </p>| <p align = "center"> Simple, quick, easy to interpret </p>|<p align = "center"> if the data is not linearly separeble, performance decreases </p>|
|<p align = "center"> KNN </p>|<p align = "center"> Easy to understand, no "real" training needed </p>|<p align = "center"> slow in predictions with a lot of data </p>|
|<p align = "center"> MLP(Neural Network)</p>|<p align = "center"> More flexible, it adapts well to complex patterns </p>| <p align = "center"> it can bre more difficult to adjust and slower </p>|


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** Used to work with arrays and matrices, and it has functions for mathematics and statistics <br> **matplotlib.pyplot:** Used to create graphs, and pyplot is specifically for generating different types of plots (e.g., bar, line, scatter).<br> **load_digits from sklearn.datasets:** Used to load a dataset of handwritten digits, which is commonly used in classification projects.<br> **sklearn.model_selection import train_test_split:** Function to split the data into two parts: Training data (Train the model) and Test data (Evaluate the model) <br> **sklearn.preprocessing import StandardScaler:** Standard scaler that normalizes the data, Mean 0, Standard deviation 1, this makes training more efficient in many algorithms <br> **sklearn.linear_model import LogisticRegression:** Import the logistic regression classifier <br> **sklearn.neighbors import KNeighborsClassifier:** Import the K-Nearest Neighbors (KNN) classifier,search for the nearest neighbors. <br> **sklearn.svm import SVC:** Import the SVM (Support Vector Machine) classifier, it is a powerful classification algorithm that seeks the best boundary to separate the classes.<br> **sklearn.neural_network import MLPClassifier:** Import a classifier based on Neural Networks (Multi-layer Perceptron - MLP), it is a simple neural network that is widely used for classification and regression.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/1.ImportLibraries.png" width="4000"/>|
**`load_digits()`:** Load the digits dataset <br> **`x`:** Images as vectors of 64 elements (8x8). <br> **`y`:** The labels of the digits(0-9)| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/2.LoadDataset.png" width="4000"/>|
Divide the data into two sets (80% for training and 20% for testing). <br> **`random_state`:** the division is always the same|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/3.SplitTrainAndTestSets.png" width="4000"/>|

**`clases`:** Contains the unique values of y(0-9) <br> **`count_classes`:** Array that indicates how many examples there are of each class | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/1.ImportLibraries.png" width="4000"/>|

**`plt.figure(figsize=(8,5))`:** Set the size of the figure (8 wide and 5 high) <br> **`plt.bar(clases, count_classes, color='skyblue')`:** Create the bar chart, the list that contains the number of examples, assign the color of the bars. <br> **`plt.xlabel('Digit')`:** Set the x-axis label <br> **`plt.ylabel('Number of examples')`:** Set the y-axis label <br> **`plt.title('Distribution of the classes')`:** Add a title to the chart. <br> **`plt.show()`:** Visualize the graph | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/1.ImportLibraries.png" width="4000"/>|

**`fig, axes = plt.subplots(2, 5, figsize=(10, 5))`:** Create a figure and a grid (2 rows and 5 columns) totaling 10 subplots, set the size (10, 5 in inches)  10 wide and 5 high <br> **`fig.suptitle("Examples of images")`:** Establish a general title <br> **`for i, ax in enumerate(axes.ravel())`:** Iterates through each of the subplots, converts the (2x5) matrix into a one-dimensional array, which makes individual access easier, and returns both the index `i` and the `ax` object in each iteration. <br> **`ax.imshow(x[i].reshape(8,8), cmap='gray')`:** it is a vector image from the dataset.  resize to an 8x8 matrix, apply a grayscale <br> **`ax.set_title(f"label: {y[i]}")`:** Assign a title to each subplot. <br> **`ax.axis('off')`:** Deactivate the axes so that the marks or values do not appear. <br> **`plt.show()`:** Show the subplots and images. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/1.ImportLibraries.png" width="4000"/>|