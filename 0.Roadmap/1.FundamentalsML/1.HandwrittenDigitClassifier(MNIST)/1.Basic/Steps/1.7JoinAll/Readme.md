<p align = "center" >
    <h1 align = "Center"> 7.Join All </h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

The objective is to integrate the entire workflow into a single efficient and clear code, facilitating its execution and scalability, to be able to consolidate all the key steps of the Machine Learning workflow for the MNIST 8x8 dataset into a single robust program, this includes:
- **Loading the Dataset:** It retrieves the data from the reduced MNIST dataset.
- **Data Exploration:** Shows dimensions and class distribution.
- **Preprocessing:** Includes normalization with StandardScaler.
- **Hyperparameter Tuning:** Use GridSearchCV to find the best parameters for the SVM model.
- **Model Training:** Use the optimal model obtained after hyperparameter tuning.
- **Model Evaluation:** Displays metrics such as accuracy, confusion matrix, and classification report.
- **Visualization:** Includes charts such as the confusion matrix and examples of predictions.



<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h3 align = "Center"> 1.True and Predict </h3>
</p>
This visualization shows 10 selected images from the test set, along with their actual labels (True) and the model's predictions (Predicted).  It serves to visually compare how well the model identifies the digits, highlighting correct predictions and possible errors in classification.

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/TrueAnd Predict.png" width="4000"/>

<p align = "center" >
    <h3 align = "Center">2.Confusion Matrix</h3>
</p>
It's a table that shows how many times the model correctly classified each number and how many times it made mistakes.
**To read it**
- The rows represent the real numbers (true labels).
- The columns represent the model's predictions.
- The values on the main diagonal are the correct predictions (true positives).
- The values outside the diagonal are errors (incorrect predictions).

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/ConfusionMatrix.png" width="4000"/>

<p align = "center" >
    <h3 align = "Center">3.Data</h3>
</p>

This program uses the reduced MNIST dataset (8x8) which contains 1797 images of digits from 0 to 9, where each image is represented as a vector of 64 elements (corresponding to an 8x8 pixel image).  The label set (y) also contains 1797 values, indicating the digit corresponding to each image.
Additionally, the distribution of classes in the dataset is shown, where each digit has a similar number of examples, thus ensuring a relatively balanced dataset.

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/Data.png" width="4000"/>


<p align = "center" >
    <h3 align = "Center">4.Data Normalized</h3>
</p>

This program uses the reduced MNIST dataset (8x8) which contains 1797 images of digits from 0 to 9, where each image is represented as a vector of 64 elements (corresponding to an 8x8 pixel image). The label set (y) also contains 1797 values, indicating the digit corresponding to each image.
Additionally, the distribution of classes in the dataset is shown, where each digit has a similar number of examples, thus ensuring a relatively balanced dataset.

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/DataNormalized.png" width="4000"/>

<p align = "center" >
    <h3 align = "Center">5.Classification Report</h3>
</p>
This report allows us to evaluate how well the model distinguishes between the different classes of the dataset.

The Classification Report provides a detailed summary of the model's performance in each class.  It includes key metrics such as:
- **Precision:** Percentage of correct predictions within the predicted positive labels.
- **Recall:** Percentage of positive examples that the model correctly identified.
- **F1-Score:** Harmonic mean between precision and recall, ideal when seeking a balance between both metrics.
- **Support:** Total number of real examples in each class.

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/ClassificationReport.png" width="4000"/>


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** For numerical data manipulation. <br> **Matplotlib and Seaborn:** To visualize the confusion matrix. <br> **Scikit-learn:** For handling the dataset, preprocessing, model training, and evaluation.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/1.ImportLibraries.png" width="4000"/>|
**`load_digits()`:** Loads the reduced MNIST dataset with 8x8 pixel images. <br> **`digits.data`:** Each image is represented as a vector of 64 elements. <br> **`digits.target`:** Contains the labels (classes from 0 to 9).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/2.LoadDataset.png" width="4000"/>|
**`x.shape`:** Shows that there are 1797 images with 64 features each.<br> **`y.shape:`** Contains 1797 labels.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/3.ExploringDimensionsDataset.png" width="4000"/>|
**`np.unique()`:** Shows the classes of the dataset (0 to 9) and how many images there are for each class.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/4.ClassesNumberExamples.png" width="4000"/>|
**`train_test_split()`:**  Divide the dataset into: <br> **`x_train`** and **`y_train`** (**80%** for training). <br> **`x_test`** and **`y_test`** (**20%** for testing).<br> **`random_state=42`:** Ensures that the split is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/5.SplitIntoTrainTestSets.png" width="4000"/>|   
**`StandardScaler()`:** <br> Scale the data to have a mean of 0 and a standard deviation of 1, improving the model's performance. <br> **`fit_transform()`** is used on the training set. <br> **`transform()`** is applied to the test set using the same parameters learned from the training set.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/6.NormalizationStandarScaler.png" width="4000"/>|
**`param_grid`:** Defines the parameters that will be evaluated. <br> **`C`:** Controls the decision margin in the SVM. <br> **`kernel`:** Define the data transformation function (linear or rbf). <br> **`GridSearchCV()`:** <br> Conduct an exhaustive search to find the best combination of hyperparameters. <br> **`cv=5`:** Use 5-fold cross-validation on the training set.<br> **`scoring='accuracy'`:** Optimize accuracy| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/7.HyperparametersTuningGRidSearch.png" width="4000"/>|
**`grid_search.best_estimator_`** selects the best model found in the hyperparameter search. <br> **`fit()`** trains the model with the training data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/8.TrainModelsEvaluate.png" width="4000"/>|
**`predict()`:** Generates predictions for the test set. <br> **`score()`:** Calculates the model's accuracy. <br> **`confusion_matrix()`:** Shows the correct and incorrect predictions by class. <br> **`classification_report()`:** Provides detailed metrics such as precision, recall, and F1-score.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/9.EvaluationMetrics.png" width="4000"/>|
**`heatmap()`:** Visualizes the confusion matrix with colors to facilitate interpretation. <br> Show which classes were most frequently misclassified.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/10.Visualization.png" width="4000"/>|
Show 10 examples from the test set with their actual and predicted labels.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.7JoinAll/Images/11.VisualizationPrecision.png" width="4000"/>|
