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

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|