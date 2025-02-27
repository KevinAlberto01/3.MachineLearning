<p align = "center" >
    <h1 align = "Center"> Data Argumentation </h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

The purpose of implementing Data Augmentation with Keras, Albumentations, and OpenCV was to improve the performance of digit classification models by generating new variations of the training images.  This helps the models to be more robust and generalize better on unseen data.
Given that the MNIST 8x8 dataset has a limited number of images, artificially expanding the dataset allows models to learn more varied patterns and avoid overfitting.  To achieve this, three different approaches were tested:

* **1.Albumentations** 
    - Advanced augmentation techniques were used with an efficient implementation for faster and more varied transformations.

* **2.Keras**
    - The ImageDataGenerator was utilized, which allows for dynamic augmentations during training, reducing the need to store augmented images.

* **3.OpenCV**
    - Basic transformations such as rotation, translation, and noise addition were applied. This method offered manual control over each transformation.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h3 align = "Center">1.Accuracy</h3>
</p>

Analyze how data augmentation through transformations such as rotations, translations, and noise affects the models' ability to generalize over the test set.


 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/Data.png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/Data.png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/Data.png" width="1000"/>|
|**<p align = "center" > Logistic Regression </p>**|**<p align = "center" > Logistic Regression </p>**|**<p align = "center" > Logistic Regression </p>**|
|**Accuracy:** 0.8306 <br> **Best Parameter:** ('C': 0.1, 'max_iter':1000)| **Accuracy:** 0.68 <br> **Best Parameter:** ('C': 0.1, 'max_iter':1000)|**Accuracy:** 0.6843 <br> **Best Parameter:** ('C': 0.1, 'max_iter':1000)|
|**<p align = "center" > K-Nearest Neighbors (KNN) </p>**|**<p align = "center" > K-Nearest Neighbors (KNN) </p>**|**<p align = "center" > K-Nearest Neighbors (KNN) </p>**|
|**Accuracy:** 0.8889 <br> **Best Parameter:** ('n_neighbors' : 7)| **Accuracy:** 0.86 <br> **Best Parameter:** ('n_neighbors' : 7)|**Accuracy:** 0.8081 <br> **Best Parameter:** ('n_neighbors' : 7)|
|**<p align = "center" > SVM </p>**|**<p align = "center" > SVM </p>**|**<p align = "center" > SVM </p>**|
|**Accuracy:** 0.9389 <br> **Best Parameter:** ('C': 10, 'kernel':'rbf')| **Accuracy:** 0.91 <br> **Best Parameter:** ('C': 10, 'kernel':1000)|**Accuracy:** 0.8401 <br> **Best Parameter:** ('C': 10, 'kernel': 'rbf')|
|**<p align = "center" > Multi-Layer Perceptron(MLP) </p>**|**<p align = "center" > Multi-Layer Perceptron(MLP) </p>**|**<p align = "center" > Multi-Layer Perceptron(MLP) </p>**|
|**Accuracy:** 0.9333 <br> **Best Parameter:** ('hidden_layer_sizes': (100,), 'max_iter':'500')| **Accuracy:** 0.97 <br> **Best Parameter:** ('hidden_layer_sizes': (100,), 'max_iter':'500')|**Accuracy:** 0.7969 <br> **Best Parameter:** ('hidden_layer_sizes': (100,), 'max_iter':'2000')|


<p align = "center" >
    <h3 align = "Center">2.Bars Graph</h3>
</p>
Show the improved accuracy of each model so you can compare them visually.

 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/Results.png" width="850"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/Results.png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/Results.png" width="1000"/>|

<p align = "center" >
    <h3 align = "Center">3.Confusion Matrix</h3>
</p>
It's a table that shows how many times the model correctly classified each number and how many times it made mistakes.
**To read it**
- The rows represent the real numbers (true labels).
- The columns represent the model's predictions.
- The values on the main diagonal are the correct predictions (true positives).
- The values outside the diagonal are errors (incorrect predictions).

###  <p align = "center" > **3.1 Logistic Regression** </p>
 
 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/ConfusionMatrix(LR).png" width="800"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/ConfusionMatrix(LR).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/ConfusionMatrix(LR).png" width="1000"/>|

###  <p align = "center" > **3.2 K-Nearest(KNN)** </p>
 
 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/ConfusionMatrix(KNN).png" width="800"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/ConfusionMatrix(KNN).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/ConfusionMatrix(KNN).png" width="1000"/>|

###  <p align = "center" > **3.3 SVM** </p>
 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/ConfusionMatrix(SVM).png" width="800"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/ConfusionMatrix(SVM).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/ConfusionMatrix(SVM).png" width="1000"/>|

###  <p align = "center" > **3.4 MLP** </p>
 
 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/ConfusionMatrix(MLP).png" width="800"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/ConfusionMatrix(MLP).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/ConfusionMatrix(MLP).png" width="1000"/>|


<p align = "center" >
    <h3 align = "Center">4.Classification Report </h3>
</p>

###  <p align = "center" > **3.1 Logistic Regression** </p>
 
 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/ReportClassification(LR).png" width="800"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/ReportClassification(LR).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/ReportClassification(LR).png" width="1000"/>|

###  <p align = "center" > **3.2 K-Nearest Neighbors(KNN)** </p>

 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/ReportClassification(KNN).png" width="800"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/ReportClassification(KNN).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/ReportClassification(KNN).png" width="1000"/>|

###  <p align = "center" > **3.3 SVM** </p>
 
 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/ReportClassification(SVM).png" width="800"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/ReportClassification(SVM).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/ReportClassification(SVM).png" width="1000"/>|
###  <p align = "center" > **3.4 MultiLayer Perceptron(MLP)** </p>
 
 **<p align = "center" > With Albumentations </p>**| **<p align = "center" > With Keras </p>**|**<p align = "center" > With OpenCV </p>**|
|------------------------|------------------------|----------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/ReportClassification(MLP).png" width="800"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/ReportClassification(MLP).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/ReportClassification(MLP).png" width="1000"/>|


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