# 1. Handwritten Digits Classifier (MNIST)

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>
Develop a classification model capable of correctly predicting the digit present in each image using Machine Learning algorithms.
## Description
Handwriting recognition is one of the most common applications of Machine Learning and Computer Vision. This problem consists of building a classification model capable of recognizing handwritten digits (from 0 to 9) from grayscale images.


<p align = "center" >
    <h2 align = "Center">💻​ Technologies Implement 💻​</h2>
</p>
Skill: Logistic regression and simple neural networks.
Description: Use Scikit-learn or TensorFlow to classify digits from the MNIST dataset.
Extra: Implement the model from scratch without advanced frameworks.

To do this, we will use the MNIST (Modified National Institute of Standards and Technology) dataset, a widely used dataset in the field of Machine Learning. It contains 60,000 training images and 10,000 test images, where each image is 28x28 pixels and represents a number from 0 to 9.


<p align = "center" >
    <h2 align = "Center">📓 Loading and Exploring (MNIST) 📓</h2>
</p>

<h2 align = "Center">🔎​ 1. Loading and Exploring (MNIST) 🔍​</h2>

* **1.Understand your data before training any model** 
* **2.Identify potential problems**
* **3.Familiarize yourself with the structure of the data**
* **4.Foundation for future steps**

| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/Histogram.png" width="700"/>| <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/Result.png" alt="Result2" width="1000"/>|
|----------------------------|----------------------------------------------------------------------------|

We have a program that show the dimensions of x, y of the dataset, the classes it has numbers of 0 to 9, and the number of examples per class, it have

<h2 align = "Center">📉​ 2. Data Preprocessing 📉</h2>
Transform the raw data into a format that is more suitable for the model to learn patterns more quickly and accurately.

* **1.Normalization / Standardization**
* **2.Conversion to vectors**

|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.2DataPreprocessing/Images/Results.png" width="850"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.2DataPreprocessing/Images/DataNormalized.png" width="1000"/>|
|----------------------------|----------------------------------------------------------------------------|

View the original images to confirm that the data you are using is reasonable and makes sense, Check the normalized numerical values to ensure that the preprocessing was done correctly.

<h2 align = "Center">🧮​ 3. Training Multiple Algorithms 🧮</h2>

Trying different models allows you to understand which one works best for your problem and how performance varies depending on the algorithm.
This process teaches you not to rely on a single algorithm and to compare methods in practice.

* **1.Logistic Regression** 
* **2.K-Nearest Neighbors(KNN)**
* **3.Support Vector Machine (SVM)**
* **4.Multi-Layer Perceptron(MLP)**

|<p align = "left"> <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/Results.png" width="800"/>|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/Data.png" alt="Result2" width="1000"/>|
|----------------------------------------|----------------------------------------------------------------------------|

<p align = "center" >
    <h2 align = "Center"> 🔎 Aspects to consider 🔍</h2>
</p> 

|Algorithm|Why would you choose it?| When to avoid it? |
|---------|------------------------|-------------------|
|<p align = "center"> svm </p>|<p align = "center"> Better accuracy, works well on small datasets </p>|<p align = "center"> Can be slow on large databasets </p>|
|<p align = "center"> Logistic Regression </p>| <p align = "center"> Simple, quick, easy to interpret </p>|<p align = "center"> if the data is not linearly separeble, performance decreases </p>|
|<p align = "center"> KNN </p>|<p align = "center"> Easy to understand, no "real" training needed </p>|<p align = "center"> slow in predictions with a lot of data </p>|
|<p align = "center"> MLP(Neural Network)</p>|<p align = "center"> More flexible, it adapts well to complex patterns </p>| <p align = "center"> it can bre more difficult to adjust and slower </p>|

In this exercise we choose the **SVM Algorithm with 97% of Acurrancy** because have better precition and its excelent to small datasets 

<h2 align = "Center">📊​ 4. Evaluation Metrics 📊​</h2>

<h2 align = "Center">📈​ 5. Optimization (Tuning & Hyperparameters)📈​</h2>

<h2 align = "Center">📑​ 6. Data Argumentation 📑 ​</h2>

<h2 align = "Center">📦​ 7. Join All 📦</h2>

<h2 align = "Center">🔧​ 8. Personalisation 🔧</h2>
 
