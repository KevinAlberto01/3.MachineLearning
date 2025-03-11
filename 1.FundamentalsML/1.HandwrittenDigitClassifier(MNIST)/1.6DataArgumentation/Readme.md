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
**numpy:** For handling matrices and arrays.<br>**matplotlib:** For plotting comparisons and confusion matrices.<br>**seaborn:** Advanced visualization for the confusion matrix. <br>**albumentations:** Library specialized in Data Augmentation for images.<br>**sklearn:** For data manipulation, scaling, classification, and metrics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/1.ImportLibraries.png" width="4000"/>|
**`load_digits()`** loads the reduced MNIST dataset (8x8 digit images). <br> **`x`** is stored in its original form as 8x8 images, which makes it easier to apply **Data Augmentation**.<br> **`y`** it contains the corresponding labels.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/2.LoadDataset.png" width="4000"/>|
This section creates a Data Augmentation pipeline that includes: <br> **`ShiftScaleRotate:`** Randomly shifts, scales, and rotates the image.<br> **`shift_limit=0.01:`** Maximum displacement of 1% in any direction. <br> **`scale_limit=0.1`:** Scale the image up to 10%.<br> **`rotate_limit=15`:** Maximum rotation of ±15°. <br> **`GridDistortion`:** Deforms the image in a grid to simulate distortions. <br> **`ElasticTransform:`** Applies an elastic transformation that stretches and deforms the image to improve the model's robustness. <br> The parameter p=0.5 indicates that each transformation has a 50% probability of being applied to each image.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/3.DefineDataArgumentation.png" width="4000"/>|
Each image in the dataset is processed. <br> Each image is subjected to the transformations defined in **`transform`**. <br> The augmented images and their respective labels are stored in lists for later conversion to arrays.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/4.ApplyArgumentationTraining.png" width="4000"/>|
**`np.array()`** converts lists into **`NumPy`** arrays to facilitate handling with Scikit-learn. <br> **`reshape(len(...), -1)`** transforms the augmented **8x8** images into vectors of **64** elements (format required by the models).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/5.ConvertNumpyArray.png" width="4000"/>|
**80%** of the data is allocated for training and **20%** for testing. <br> **`random_state=42`** ensures that the split is reproducible in each execution.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/6.DivideDatasetIntoTrainingTesting.png" width="4000"/>|
**`StandardScaler()`** adjusts the data so that each feature has a **mean of 0** and a **standard deviation of 1**, which improves the performance of the models.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/7.NormalizeData.png" width="4000"/>|
The **hyperparameters** that will be optimized using **`GridSearchCV`** are defined. <br> Each model is iterated through and **`GridSearchCV`** is applied to find the best parameters. <br> The best model is saved in **`best_models`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/8.DefineModelsWithHyperparametersTuning.png" width="4000"/>|
A bar graph is generated that compares the performance (accuracy) of each optimized model.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/9.VisualizationAccuracyComparison.png" width="4000"/>|
**`Seaborn`** is used to visually display classification errors.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/10.ShowConfusionMatrixEachModel.png" width="4000"/>|
A report is generated that includes metrics such as: <br> **Precision** <br> **Recall** <br> **F1-score** <br> Final message indicating that the entire workflow has been completed successfully.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Albumentations)/11.ClassificationReport.png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Open CV 💻</h4>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**cv2 (OpenCV):** To apply Data Augmentation through visual transformations. <br> **numpy:** Handling arrays and mathematical operations. <br> **matplotlib** and **seaborn:** For visualization of graphs and confusion matrices. <br> **sklearn:** For data manipulation, scaling, classification, and metrics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/1.LibraryImportation.png" width="4000"/>|
**Rotation:** A random rotation between **-10°** and **10°** is applied. <br> **Translation:** A small random shift of **-1** to **1** pixel is applied. <br> **Gaussian Noise:** Noise with a low standard deviation (std=5) is added to simulate slight imperfections in the image. <br> The use of slight transformations improves the **model's robustness** without distorting the images too much.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/2.DataAugmentationFunctionOpencv.png" width="4000"/>|
The MNIST dataset is loaded with images in their original form **(8x8)** to facilitate visual manipulation.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/3.LoadDataset.png"/>|
Each image in the dataset is processed, and the **`apply_augmentation()`** function is applied to it.<br>The augmented images and their labels are stored in lists to later convert them into **NumPy arrays**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/4.ApplyDataAugmentationOpenCV.png" width="4000"/>|
**`reshape()`** converts **8x8** images into vectors of **64** elements. <br> **`/ 255.`** scales the pixel values to the range [0, 1] to stabilize the training.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/5.ConvertNumpyArrayandNormalize.png" width="4000"/>|
The original and augmented images are combined to increase the size of the dataset, improving the model's generalization.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/6.CombineOriginalImagesAugmented.png" width="4000"/>|
The dataset is divided into **80%** for training and **20%** for testing.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/7.SplitDatasetTrainingTest.png" width="4000"/>|
**`StandardScaler()`** normalizes the data to have a **mean of 0** and a **standard deviation of 1**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/8.ScaleData.png" width="4000"/>|
The hyperparameter search is defined with **`GridSearchCV`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/9.DefineModelsAdjustHyperparameters.png" width="4000"/>|
**`GridSearchCV`** is used to find the best hyperparameters through cross-validation. <br> The best model is stored in **`best_models`**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/10.Train%26EvaluateModels.png" width="4000"/>|
A bar graph is generated comparing the obtained accuracies.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/11.ModelComparisonVIsual.png" width="4000"/>|
The classification results of each model are visualized to identify errors.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/12.ShowConfusionMatrix.png" width="4000"/>|
The report details the Precision, Recall, and F1-score metrics.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(OpenCV)/13.CLassificationReport.png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Keras 💻</h4>
</p> 

<p align = "center" >
    <h5 align = "Center"> 1.KNN </h5>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**tensorflow and keras:** To build and train the model. <br> **numpy:** To handle arrays and perform numerical calculations. <br> **matplotlib and seaborn:** For visualizing graphs and confusion matrices. <br> **albumentations:** To apply Data Augmentation to images. <br> **sklearn:** To split the dataset and evaluate the model's performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/1.ImportLibraries.png" width="4000"/>|
The MNIST 8x8 dataset is **loaded**. <br> **`x`** contains the images (each **8x8** pixels).<br>**`y`** contains the labels (digits from 0 to 9).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/2.LoadData.png" width="4000"/>|
**`ShiftScaleRotate`:** Performs small rotations, scalings, and shifts to simulate slight variations in the images. <br> **`GridDistortion`:** Deforms the image with a grid, ideal for simulating variations in the position of the digits. <br> **`ElasticTransform`:** Applies an elastic transformation to slightly distort the images.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/3.DataAugmentation.png" width="4000"/>|
Each image in the dataset is processed, the transformation is applied, and the augmented images are stored along with their labels.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/4.ApplyDataAugmentation.png" width="4000"/>|
**`reshape()`** converts each image into a 3D matrix of shape (8, 8, 1) so that Keras can process it. <br> **`to_categorical()`** converts the labels into one-hot encoding format, which is the required format for multi-class classification.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/5.PrepareData.png" width="4000"/>|
The dataset is divided into: <br> **80%** for training. <br> **20%** for testing.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/6.DivisionDataset.png" width="4000"/>|
The pixel values are normalized to the range **`[0, 1]`** by dividing by 16 (the maximum value of the MNIST 8x8 dataset).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/7.NormalizationData.png" width="4000"/>|
**`Flatten()`** converts the 8x8 images into a vector of 64 elements. <br> **`Dense(64, activation='relu')`** adds a hidden layer with 64 neurons. <br> **`Dense(10, activation='softmax')`** adds an output layer to classify the 10 digits. <br> **`categorical_crossentropy`** is the appropriate loss function for classification problems| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/8.ModelDefinition.png" width="4000"/>|
**`model.fit()`** is the method that trains the model. <br> Its key parameters are:
<br> **`x_train and y_train`:** Training data. <br> **`validation_data=(x_test, y_test)`:** <br> Provides the validation data to evaluate the model's performance after each epoch. <br> **history** stores the training history, including accuracy and loss for each epoch.  This is useful if you want to plot the evolution of the training.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/9.ModelTraining.png" width="4000"/>|
**`model.evaluate()`** evaluates the model's performance on the test data. <br> **`test_loss`:** Measures the model's error on the test set. <br> **`test_acc`:** Represents the model's accuracy on the test set (value between 0 and 1). <br> The value shown with **`print()`** indicates how well the model generalized after training.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/10.ModelEvaluation.png" width="4000"/>|
**`model.predict()`** generates the model's predictions on the test data. <br> **`y_pred`** will contain probabilities for each class (output of the model with softmax). <br> **`np.argmax()`** is used to convert those probabilities into the predicted classes: <br> **`axis=1`**  selects the index of the class with the highest probability in each row (each image). <br> **`y_pred_classes`** are the predicted classes. <br> **`y_test_classes`** are the actual classes (converted from one-hot encoding to numerical labels).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/11.Prediction.png" width="4000"/>|
**`confusion_matrix()`** generates a matrix that shows: <br> **`Rows`:** Real classes.
**`Columns`:** Predicted Classes. <br> Each cell indicates how many times the model classified correctly or made a mistake. <br> **`sns.heatmap()`** is used to visualize the confusion matrix clearly and visually. <br> The argument **`annot=True`** adds the numbers in each cell. <br> The argument **`cmap='Blues'`** applies a blue gradient to highlight the highest values. <br> In this matrix, the darker the blue on the main diagonal, the better the model's performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/12.ConfusionMatrix.png" width="4000"/>|
**`classification_report()`** generates a detailed report that includes: <br> **`Precision`:** Percentage of correct predictions for each class. <br> **`Recall`:** The model's ability to correctly find all instances of a class. <br> **`F1-Score`:** Harmonic mean between precision and recall (better when it approaches 1). <br> **`Support`:** Total number of samples in each class.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/KNN/13.ClassificationMatrix.png" width="4000"/>|

<p align = "center" >
    <h5 align = "Center"> 2.Lineal Regression (LR) </h5>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**tensorflow:** It is a widely used machine learning library, especially for deep learning tasks. <br> **keras:** High-level API integrated into TensorFlow that makes it easier to create neural networks. <br> **layers:** Module that allows building the model layers in a modular and simple way. <br> **numpy:** Library for efficient manipulation of multidimensional arrays. <br> It is used to handle images, matrices, and perform mathematical calculations. <br> **matplotlib.pyplot:** Library for creating plots, ideal for visualizing results such as the confusion matrix or precision curves. <br> **albumentations:** Library specialized in Data Augmentation for images. <br> It is efficient and designed to improve the performance of models in computer vision. <br> **os:** Allows interaction with the operating system, such as handling file paths or configuring environments. <br> **load_digits:** Loads the Digits dataset, which contains images of handwritten digits (8x8 pixels). <br> **train_test_split:** Splits the dataset into training and test sets. <br> **StandardScaler:** Scales the data to improve model performance, especially in algorithms sensitive to scale. <br> **classification_report:** Generates a report with metrics such as precision, recall, and F1-score. <br> **confusion_matrix:** Creates the confusion matrix to evaluate the model's performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/1.ImportLibraries.png" width="4000"/>|
This section allows TensorFlow to use the GPU, but in a way that it only consumes the memory it needs. <br> If no GPU is available, this block is ignored without affecting the code.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/2.SettingTensoflowGPU.png" width="4000"/>|
The Digits dataset from scikit-learn is loaded. <br> **`x`** stores the images (each of size 8x8). <br> **`y`** contains the corresponding labels (digits from 0 to 9).|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/3.LoadDataset.png" width="4000"/>|
**Data Augmentation** is applied using **Albumentations**: <br> **`ShiftScaleRotate`:** Moves, scales, and rotates the image.<br> **`GridDistortion`:** Applies a grid-type distortion to simulate deformations. <br> **`ElasticTransform`:** Generates an elastic effect on the image. <br> The **`p=0.5`** value indicates that each transformation will be applied with a 50% probability.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/4.DataAugmentatinoAlbumentations.png" width="4000"/>|
Iterate over the images and labels of the original dataset. <br> Each image is transformed and stored in the **augmented_images** list. <br> The corresponding labels are stored in **augmented_labels**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/5.DataPreparation.png" width="4000"/>|
**`reshape()`** changes the shape of the images to fit the Keras input **(8x8x1)**. <br> **`to_categorical()`** converts the labels into one-hot encoding format for the model.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/6.DataConversation.png" width="4000"/>|
The data is split into a **training** set **(80%)** and a **test** set **(20%)**.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/7.%20SplitDatasetTrainning.png" width="4000"/>|
Divide each pixel by **16** so that the values are in the range **[0, 1]**. <br>This improves numerical stability and model performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/8.DataNormalization.png" width="4000"/>|
**`Flatten()`** converts the 8x8 image into a vector of 64 elements.  <br> **`Dense(10, activation='softmax')`** creates the output layer with 10 neurons (one for each class). <br> **`softmax`** converts the outputs into probabilities. <br> **`softmax`** converts the outputs into probabilities. <br> The model is compiled with: <br> **`loss="categorical_crossentropy"`**: Loss function for multiclass classification problems. <br> **`optimizer="adam"`:** Efficient optimizer that automatically adjusts the learning rate. <br> **`metrics=["accuracy"]`**: To measure the model's accuracy.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/9.DefineModelLogistic.png" width="4000"/>|
Train the model for **20 epochs** in batches of **32 images** each. <br> The performance is evaluated on the test set using the **validation_data argument.**| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/10.CreateTrainModel.png" width="4000"/>|
Evaluate the **loss (`test_loss`)** and **accuracy (`test_acc`)** on the test set.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/11.EvaluationModel.png" width="4000"/>|
Generate the model predictions.<br> **`np.argmax()`** converts probabilities into predicted and actual labels.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/12.Prediction.png" width="4000"/>|
Show the confusion matrix to evaluate performance class by class.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/13.ConfusiionMatrix.png" width="4000"/>|
Provide detailed metrics such as **precision, recall, and F1-score**.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/14.ClassificationReport.png" width="4000"/>|
The confusion matrix is visualized using **`seaborn.heatmap()`** to facilitate interpretation.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/LR/15.ConfusionMatrixGraph.png" width="4000"/>|

<p align = "center" >
    <h5 align = "Center"> 3.MLP </h5>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** For handling arrays and numerical data. <br> **Matplotlib & Seaborn:** To visualize the confusion matrix.<br> **TensorFlow/Keras:** To create and train the MLP model. <br> **sklearn:** For dataset handling, data splitting, and model evaluation.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/MLP/1.ImportLibraries.png" width="4000"/>|
**`digits.images`:** Contains the images of the digits in the form of 8x8 matrices. <br> **`digits.target`:** Corresponding labels (digits from 0 to 9).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/MLP/2.LoadData.png" width="4000"/>|
**`reshape()`:** Resizes each image from (8, 8) to (8, 8, 1).  <br> This adds an additional dimension that represents the color channel (in this case, grayscale). <br> **`/ 16.0`:** Normalizes the images so that their values are in the range [0, 1].  Since the original data ranges from 0 to 16, it is divided by 16.<br> **`to_categorical()`:** Converts the labels **(0-9)** into **one-hot encoding** to fit the model's output.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/MLP/3.NormalizeRendimension.png" width="4000"/>|
**`train_test_split()`:**  Divide the dataset into: <br> **`x_train and y_train`:** Training data (80% of the total). <br> **`x_test and y_test`:** Test data (20% of the total). <br> **`random_state=42`:** Ensures that the split is reproducible.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/MLP/4.DataserS%E1%B9%95lit.png" width="4000"/>|
**`Flatten()`:** Flattens 8x8 images into 64-element vectors to be processed in dense layers. <br> **`Dense()`:** Fully connected layer. <br> **128** and **64:** Neurons in the hidden layers to learn complex patterns. <br> **`activation='relu'`:** Activates only positive values, ideal for deep networks. <br> **`activation='softmax'`:** Assigns probabilities to each class in the output. <br> **compile():** <br> **`loss='categorical_crossentropy'`:** Loss function suitable for multiclass problems. <br> **`optimizer='adam'`:** Efficient algorithm to optimize the model. <br> **`metrics=['accuracy']`:** Measures the model's performance in accuracy.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/MLP/5.ConstructMLPModel.png" width="4000"/>|
**`fit()`:** Train the model. <br> **`epochs=20`:** The model will make 20 complete passes through the dataset. <br> **`batch_size=32`:** Train in batches of 32 samples for greater efficiency. <br> **`validation_data=(x_test, y_test)`:** Evaluates the model on the test data during training to verify performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/MLP/6%2CTrainMLPModel.png" width="4000"/>|
**`predict()`:** Predicts the classes in the test set. <br> **`np.argmax()`:** Takes the index with the highest probability in each prediction to obtain the predicted class.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/MLP/7.Prediction.png" width="4000"/>|
**Show the key metrics:** <br> **`Precision`:** How precise was the model for each class. <br> **`Recall`:** How well did the model identify the correct classes <br> **`F1-Score`:** Average between precision and recall.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/MLP/8.Classification.png" width="4000"/>|
**`confusion_matrix()`:** Evaluates the number of correct and incorrect predictions per class. <br> **`sns.heatmap()`:** Visualizes the confusion matrix with colors to facilitate interpretation.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/MLP/9.ConfusionMatrix.png" width="4000"/>|

<p align = "center" >
    <h5 align = "Center"> 4.SVM </h5>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** For handling numerical data. <br> **Matplotlib & Seaborn:** For visualizing the confusion matrix. <br> **TensorFlow/Keras:** To build, compile, and train the model. <br> **Scikit-learn:** For dataset management and model evaluation.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/1.ImportLibraries.png" width="4000"/>|
**`digits.images`:** Contiene las imágenes de los dígitos en matrices de 8x8. <br> **`reshape()`:** Agrega una dimensión adicional para que cada imagen tenga forma (8, 8, 1), adaptándose al formato que espera Keras. <br> **`to_categorical()`:** Convierte las etiquetas en formato **one-hot encoding** para que el modelo trabaje mejor en problemas de clasificación multiclase.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/2.LoadDataset.png" width="4000"/>|
**`train_test_split()`** splits the dataset into: <br> **`x_train`** and **`y_train`** (**80%** for training). <br> **`x_test`** and **`y_test`** (**20%** for testing). <br> **`stratify=y`:** Ensures that the class distribution remains balanced in both sets, avoiding biases.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/3.DivideDataset.png" width="4000"/>|
Divide by **16.0** because the original values of the dataset range from **0** to **16**, thus normalizing them in the **[0, 1]** range.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/4.NormalizationData.png" width="4000"/>|
**`np.argmax()`:** Converts the one-hot encoding into its respective original labels. <br> **`np.unique()`:** Shows the number of examples per class in each set.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/5.VerificationDistributionClasses.png" width="4000"/>|
**`Flatten()`:** Flattens 8x8 images into 64-element vectors. <br> **`Densa(10, activación='softmax')`:** <br> **10:** Because there are 10 possible classes (digits from 0 to 9). <br> **softmax:** Transforms the outputs into probabilities for each class. <br> Although it seems like a basic model, this behavior mimics the linear decision function of an SVM for multiclass classification.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/6.ConstructionModelKetas.png" width="4000"/>|
**`loss='categorical_crossentropy'`:** For multiclass classification problems. <br> **`optimizer='adam'`:** Efficient and robust optimizer. <br> **`metrics=['accuracy']`:** Measures the model's accuracy during training.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/7.CompilationModel.png" width="4000"/>|
**`epochs=20`:** The model will train for 20 complete iterations over the data. <br> **`batch_size=32`:** It will train in batches of 32 samples. <br> **`validation_data=(x_test, y_test)`:** Evaluates performance on the test set during training.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/8.TrainingModel.png" width="4000"/>|
**`predict()`:** Predicts the probabilities for each class in the test set. <br> **`np.argmax()`:** Converts the probabilities into class labels.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/9.EvaluationModel.png" width="4000"/>|
**`confusion_matrix()`:** Shows the correct and incorrect predictions by class. <br> **`sns.heatmap()`:** Visualizes the results clearly with colors.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/10.MatrixConfusion.png" width="4000"/>|
The **`classification_report()`** provides: <br> **`Precision`:** How accurate were the predictions for each class. <br> **`Recall`:** How well did you identify the correct classes .<br> **`F1-Score`:** Average between precision and recall.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.6DataArgumentation/Images(Keras)/SVM/11.ReportClassification.png" width="4000"/>|