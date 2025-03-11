<p align = "center" >
    <h1 align = "Center"> 5.Optimization Tuning & Hyperparameters </h1>
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
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h3 align = "Center">1.Hyperparameter</h3>
</p>

After optimizing the hyperparameters of each model using GridSearchCV, they were evaluated on the test set.
 **Acurracy Without Hyperparameter**| **Acurracy with Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/Data.png" width="700"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/Data.png" width="2000"/>|

<p align = "center" >
    <h3 align = "Center">2.Bars Graph</h3>
</p>
Show the improved accuracy of each model so you can compare them visually.

 **Bar Graph Before**| **Bar Graph After**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/Results.png" width="700"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/Results.png" width="750"/>|

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
 
| **Without Hyperparameter**| **With Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ConfusionMatrix(LR).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/ConfusionMatrix(LR).png" width="1000"/>|

###  <p align = "center" > **3.2 K-Nearest(KNN)** </p>
 
| **Without Hyperparameter**| **With Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ConfusionMatrix(KNN).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/ConfusionMatrix(KNN).png" width="1000"/>|

###  <p align = "center" > **3.3 SVM** </p>
 
| **Without Hyperparameter**| **With Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ConfusionMatrix(SVM).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/ConfusionMatrix(SVM).png" width="1000"/>|

###  <p align = "center" > **3.4 MLP** </p>
 
| **Without Hyperparameter**| **With Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ConfusionMatrix(MLP).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/ConfusionMatrix(MLP).png" width="1000"/>|


<p align = "center" >
    <h3 align = "Center">4.Classification Report </h3>
</p>

###  <p align = "center" > **3.1 Logistic Regression** </p>
 
| **Without Hyperparameter**| **With Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ReportClassification(LR).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/ReportClassification(LR).png" width="1000"/>|

###  <p align = "center" > **3.2 K-Nearest Neighbors(KNN)** </p>
 
| **Without Hyperparameter**| **With Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ReportClassification(KNN).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/ReportClassification(KNN).png" width="1000"/>|

###  <p align = "center" > **3.3 SVM** </p>
 
| **Without Hyperparameter**| **With Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ReportClassification(SVM).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/ReportClassification(SVM).png" width="1000"/>|

###  <p align = "center" > **3.4 MultiLayer Perceptron(MLP)** </p>
 
| **Without Hyperparameter**| **With Hyperparameter**|
|------------------------|------------------------| 
|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.4EvaluationMetrics/Images/ReportClassification(MLP).png" width="1000"/>|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.5Optimization(Tuning%26Hyperparameters)/Images/ReportClassification(MLP).png" width="1000"/>|


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**numpy:** To handle numerical arrays efficiently. <br> **matplotlib.pyplot:** To plot the comparison of accuracy and confusion matrices. <br> **seaborn:** To visualize the confusion matrix with a clear and attractive style. <br> **load_digits:** Load the reduced MNIST dataset (8x8 digit images). <br> **train_test_split:** Split the dataset into training and test sets. <br> **GridSearchCV:** It allows performing hyperparameter search to find the best configuration for each model.<br> **StandardScaler:** Normalizes the data to improve model performance. <br> Models: <br> LogisticRegression, KNeighborsClassifier, SVC, MLPClassifier: Classification models. <br> Metrics: <br> **confusion_matrix:** To calculate the confusion matrix. <br> **classification_report:** To obtain a detailed summary of the performance of each model.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/1.ImportLibraries.png" width="4000"/>|
**`load_digits()`:** Loads the reduced MNIST dataset, which contains 8x8 pixel grayscale images of handwritten digits. <br> **`x`:** Contains the images in the form of 64-element vectors (each element represents a pixel). <br> **`y`:** Contains the labels (values between 0 and 9).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/2.LoadDataset.png" width="4000"/>|
**`train_test_split()`:**  Divide the dataset into: <br> **80%** for **training**. <br> **20%** for testing. <br> **`random_state = 42`:** Sets the seed to ensure that the split is always the same, guaranteeing reproducible results.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/3.DivideDatasetIntoTraining.png" width="4000"/>|
**`StandardScaler()`:** Scales the data so that each feature has: <br> Media = 0 <br> Standard deviation = 1 <br> **`fit_transform(x_train)`:** Fits the scaler to the training data and then transforms it. **`transform(x_test)`:** Applies the transformation to the test data using the parameters obtained from the training set.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/4.NormalizeData.png" width="4000"/>|
Each model has a set of hyperparameters that will be optimized using **`GridSearchCV`:**  <br> **Logistic Regression:** <br> **`C`** controls the regularization (higher values reduce the penalty). <br> **`max_iter`** defines the maximum number of iterations. <br> **`KNN`:** <br> **`n_neighbors`** specifies the number of neighbors to consider. <br> **SVM:** <br> **`C`** controls the penalty. <br> **`kernel`** defines the function of the hyperplane (linear or RBF). <br> **MLP (Neural Network):** <br> **`hidden_layer_sizes`** defines the number of neurons in the hidden layer. <br> **`max_iter`** specifies the maximum number of iterations for the model to converge.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/5.DefineModelsWithH.png" width="4000"/>|
A loop is created that iterates through each model and its respective hyperparameters. <br> **Inside the loop:** <br> **`GridSearchCV()`:** Performs the search for the best hyperparameters with 3-fold cross-validation. <br> **`grid_search.best_estimator_`:** Stores the best model found for each algorithm.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/6.OptimizeModels.png" width="4000"/>|
A loop is created that iterates through each model and its respective hyperparameters. <br> Inside the loop: <br> **`GridSearchCV()`:** Performs the search for the best hyperparameters with 3-fold cross-validation. <br> **`grid_search.best_estimator_`**: Stores the best model found for each algorithm.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/7.VisualizationComparationAccuracy.png" width="4000"/>|
Show a report with metrics such as precision, recall, and F1-score for each class.<br> A message is printed indicating that the optimization has been completed.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.1LoadingAndExploring(MNIST)/Images/8.ClassificationReport.png" width="4000"/>|