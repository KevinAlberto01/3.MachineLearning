<p align = "center" >
    <h1 align = "Center"> 6.Data Argumentation </h1>
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
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 
In this case, three different programs were used because the models were making many errors, but they will be shown below with each explanation.

<p align = "center" >
    <h4 align = "Center"> Albumentations 💻</h4>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**numpy:** For handling matrices and arrays.<br>**matplotlib:** For plotting comparisons and confusion matrices.<br>**seaborn:** Advanced visualization for the confusion matrix. <br>**albumentations:** Library specialized in Data Augmentation for images.<br>**sklearn:** For data manipulation, scaling, classification, and metrics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/1.ImportLibraries.png" width="4000"/>|
**`load_digits()`** loads the reduced MNIST dataset (8x8 digit images). <br> **`x`** is stored in its original form as 8x8 images, which makes it easier to apply **Data Augmentation**.<br> **`y`** it contains the corresponding labels.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/2.LoadDataset.png" width="4000"/>|
This section creates a Data Augmentation pipeline that includes: <br> **`ShiftScaleRotate:`** Randomly shifts, scales, and rotates the image.<br> **`shift_limit=0.01:`** Maximum displacement of 1% in any direction. <br> **`scale_limit=0.1`:** Scale the image up to 10%.<br> **`rotate_limit=15`:** Maximum rotation of ±15°. <br> **`GridDistortion`:** Deforms the image in a grid to simulate distortions. <br> **`ElasticTransform:`** Applies an elastic transformation that stretches and deforms the image to improve the model's robustness. <br> The parameter p=0.5 indicates that each transformation has a 50% probability of being applied to each image.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/3.DefineDataArgumentation.png" width="4000"/>|
Each image in the dataset is processed. <br> Each image is subjected to the transformations defined in **`transform`**. <br> The augmented images and their respective labels are stored in lists for later conversion to arrays.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/4.ApplyArgumentationTraining.png" width="4000"/>|
**`np.array()`** converts lists into **`NumPy`** arrays to facilitate handling with Scikit-learn. <br> **`reshape(len(...), -1)`** transforms the augmented **8x8** images into vectors of **64** elements (format required by the models).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/5.ConvertNumpyArray.png" width="4000"/>|
**80%** of the data is allocated for training and **20%** for testing. <br> **`random_state=42`** ensures that the split is reproducible in each execution.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/6.DivideDatasetIntoTrainingTesting.png" width="4000"/>|
**`StandardScaler()`** adjusts the data so that each feature has a **mean of 0** and a **standard deviation of 1**, which improves the performance of the models.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/7.NormalizeData.png" width="4000"/>|
The **hyperparameters** that will be optimized using **`GridSearchCV`** are defined. <br> Each model is iterated through and **`GridSearchCV`** is applied to find the best parameters. <br> The best model is saved in **`best_models`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/8.DefineModelsWithHyperparametersTuning.png" width="4000"/>|
A bar graph is generated that compares the performance (accuracy) of each optimized model.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/9.VisualizationAccuracyComparison.png" width="4000"/>|
**`Seaborn`** is used to visually display classification errors.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/10.ShowConfusionMatrixEachModel.png" width="4000"/>|

A report is generated that includes metrics such as: <br> **Precision** <br> **Recall** <br> **F1-score** <br> Final message indicating that the entire workflow has been completed successfully.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/tree/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/11.ClassificationReport.png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Open CV 💻</h4>
</p> 
|Pseudocode| Image of the program|
|----------|---------------------|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

|<img src = ""/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|


<p align = "center" >
    <h4 align = "Center"> Keras 💻</h4>
</p> 
|Pseudocode| Image of the program|
|----------|---------------------|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

|<img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|

| <img src = "" width="4000"/>|