<p align = "center" >
    <h1 align = "Center"> 4.Evaluation Mectris </h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

The purpose of this stage is to analyze how well each model is performing in classifying the handwritten numbers from the MNIST 8x8 dataset.  We not only want to know the percentage of times it gets it right (accuracy), but also where it makes mistakes and how well it handles each individual number.

* **1.Accuracy (General Precision)** 
    - Indicate what percentage of the predictions were correct.
    - Limited because it does not show where the model makes mistakes.

* **2.Confusion Matrix**
    - Show how many times the model was correct and in which cases it was wrong.  
    - It is useful for identifying error patterns.

* **3.Classification Report**
    - Provide detailed metrics (precision, recall, and F1-score) for each number from 0 to 9.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h3 align = "Center">1.Accuracy (General Precision)</h3>
</p>

Provide detailed metrics (precision, recall, and F1-score) for each number from 0 to 9.
<p align = "center" >
    <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/Data.png" width="600"/>
</p>

<p align = "center" >
    <h3 align = "Center">2.Bars Graph</h3>
</p>
Show the accuracy of each model so you can compare them visually.
<p align = "center" >
    <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/Results.png" width="500"/>
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
**NumPy:** Used to work with arrays and matrices, and it has functions for mathematics and statistics <br> **matplotlib.pyplo:** To create graphs, in this case, those for comparing results and confusion matrices. <br> **seaborn:** For a more attractive and detailed visualization of confusion matrices. <br> **sklearn.datasets.load_digits:** To load the handwritten digits dataset (reduced MNIST).<br> **train_test_split:** To split the dataset into training and test subsets. <br> **StandardScaler:** To normalize the data and improve the model's performance. <br> **LogisticRegression:** For classification using logistic regression <br> **KNeighborsClassifier:** Implements the K-Nearest Neighbors algorithm. <br> **SVC:** For classification using support vector machines with a linear kernel.<br> **MLPClassifier:** For classification with a multilayer perceptron neural network. <br> **confusion_matrix:** To calculate and display the confusion matrix.<br> **classification_report:** To obtain a detailed report of the classification metrics.<br> **accuracy_score:** To calculate the accuracy of the model.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics(MNIST)/Images/1.ImportLibraries.png" width="4000"/>|
**`load_digits()`:** Loads the digit image dataset, where each image is 8x8 pixels and is represented as a vector of 64 features (pixels).<br> **`x`:** It contains the images in a matrix format with 64 features per image. <br> **`y:`** It contains the labels, which are the values between 0 and 9, representing the digits.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics(MNIST)/Images/2.LoadDataset.png" width="4000"/> 
**`train_test_split()`:**  Divide the data into training and test subsets.  In this case, 20% of the data is used for testing, and 80% for training.  In this case, 20% of the data is used for testing, and 80% for training. <br> **`random_state=42:`**  **`random_state=42:`**  A seed is set to ensure that the split is reproducible.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics(MNIST)/Images/3.DivideDataset.png" width="4000"/>|
**`StandardScaler()`**: Normalizes the data so that each feature has a mean of 0 and a standard deviation of 1, improving the performance of many machine learning algorithms. <br> **`fit_transform(x_train):`** Fits the scaler to the training data and transforms it. <br> **`transform(x_test)`:**  Apply the same transformation to the test data, using the parameters calculated from the training data.| <img src ="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics(MNIST)/Images/4.NormalizeData.png" width="4000"/>|
**Logistic Regression:** max_iter = 5000 allows for more iterations to converge. <br>  **K-Nearest Neighbors (KNN):** n_neighbors = 3 means that the model considers the 3 nearest neighbors to make a prediction. <br>  **Support Vector Machines (SVM):** With a linear kernel for classification. <br> **Multilayer Perceptron (MLP):** Neural network with one hidden layer of 50 neurons.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics(MNIST)/Images/5.DefineModels.png" width="4000"/>|
Each of the models is trained and evaluated: <br> **`Training (model.fit())`:** Trains the model with the training data (x_train, y_train). <br> **`Prediction (model.predict())`:** Makes predictions with the test data (x_test). <br> **`Accuracy (model.score())`:** Measures the model's precision on the test data. <br> **`Confusion Matrix`:** Evaluates how the model classifies correctly (diagonal) and incorrectly (off-diagonal). <br> **`Classification Report`:** Provides metrics such as precision, recall, F1-score for each class.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics(MNIST)/Images/6.TrainingEvaluateModels.png" width="4000"/>|
A bar chart is created to compare the accuracy of the trained models.<br> **`plt.bar()`:** Creates a bar chart with the accuracy results.<br>The confusion matrix of each model is displayed using seaborn for clearer visualization. <br> The confusion between classes is analyzed, indicating which digits were most confused by each model.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics(MNIST)/Images/8.Comparation.png" width="4000"/>|
The classification report for each model is printed, which includes detailed metrics such as precision, recall, and F1-score for each class (digit).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics(MNIST)/Images/9.ClassificationReport.png" width="4000"/>|

