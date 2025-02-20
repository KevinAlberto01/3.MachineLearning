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
|----------------------------|----------------------------------------------------------------------------|

Percentage of correct answers in the test set (the closer to 1 or 100%, the better).
<p align="center">
  <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/Data.png" alt="Result2" width="1000"/>
</p>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** Used to work with arrays and matrices, and it has functions for mathematics and statistics <br> **Pandas:** Used to manipulate and analyze data, particularly with DataFrames (tables of rows and columns).<br> **matplotlib.pyplot:** Used to create graphs, and pyplot is specifically for generating different types of plots (e.g., bar, line, scatter).<br> **load_digits from sklearn.datasets:** Used to load a dataset of handwritten digits, which is commonly used in classification projects.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/1.DeclareteLibraries.png" width="4000"/>|
**`digits`:** loads a dataset of handwritten digit images, where you find images of numbers 0 to 9 in black and white, each with 8x8 pixels. Each image represents a number (digit) and has a label indicating which number it is. <br> **`x`:** Contains images (numeric format), for each 8x8 image it is flattened into a 64 array, each value represents the pixel intensity (0 = black, 16 = white). <br> **`y`:** contains the labels (real numbers of the images), each element is a number from 0 to , representing which digit each image is| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/2.LoadDataset.png" width="4000"/>|
**`x.shape`:** Returns the dimension of x(1797, 64), it has 1797 images, and each image has 64 values (8x8 pixels). <br> **`y.shape`:** Returns the dimension of y (1797), there are 1797 labels per image, each label is the number it represents (0-9).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/3.ExploringDimensionsDataSet.png" width="4000"/>|
**`clases`:** Contains the unique values of y(0-9) <br> **`count_classes`:** Array that indicates how many examples there are of each class | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/4.CheckClassesNumberExamples.png" width="4000"/>|
**`plt.figure(figsize=(8,5))`:** Set the size of the figure (8 wide and 5 high) <br> **`plt.bar(clases, count_classes, color='skyblue')`:** Create the bar chart, the list that contains the number of examples, assign the color of the bars. <br> **`plt.xlabel('Digit')`:** Set the x-axis label <br> **`plt.ylabel('Number of examples')`:** Set the y-axis label <br> **`plt.title('Distribution of the classes')`:** Add a title to the chart. <br> **`plt.show()`:** Visualize the graph | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/5.ViewDistributionClasses.png" width="4000"/>|
**`fig, axes = plt.subplots(2, 5, figsize=(10, 5))`:** Create a figure and a grid (2 rows and 5 columns) totaling 10 subplots, set the size (10, 5 in inches)  10 wide and 5 high <br> **`fig.suptitle("Examples of images")`:** Establish a general title <br> **`for i, ax in enumerate(axes.ravel())`:** Iterates through each of the subplots, converts the (2x5) matrix into a one-dimensional array, which makes individual access easier, and returns both the index `i` and the `ax` object in each iteration. <br> **`ax.imshow(x[i].reshape(8,8), cmap='gray')`:** it is a vector image from the dataset.  resize to an 8x8 matrix, apply a grayscale <br> **`ax.set_title(f"label: {y[i]}")`:** Assign a title to each subplot. <br> **`ax.axis('off')`:** Deactivate the axes so that the marks or values do not appear. <br> **`plt.show()`:** Show the subplots and images. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/6.ViewExample.png" width="4000"/>|