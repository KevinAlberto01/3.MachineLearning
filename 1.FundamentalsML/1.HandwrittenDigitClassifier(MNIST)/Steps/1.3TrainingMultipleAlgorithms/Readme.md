<p align = "center" >
    <h1 align = "Center"> 3.Training Multiple Algorithms</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

Trying different models allows you to understand which one works best for your problem and how performance varies depending on the algorithm.
This process teaches you not to rely on a single algorithm and to compare methods in practice.

* **1.Logistic Regression** 
    - Basic but effective 

* **2.K-Nearest Neighbors(KNN)**
    - Sensitive to Neighbors and data

* **3.Support Vector Machine (SVM)**
    - Tens to be robust for this type of problems 

* **4.Multi-Layer Perceptron(MLP)**
    - Can Perform very well, but it depends on the training

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
**`StandardScaler()`:** Normalize the data <br> **`x_train`:** Calculate mean and standard deviation <br> **`x_test`:** Using the same mean and standar deviation <br> Normalizing improves the performance of many algorithms| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/4.NormalizeData.png" width="4000"/>|
**`Logistic Regression`:** Simple classification based on probabilities,**`max_iter=5000 Logistic`** regression may need more iterations to converge.<br> **`K-Nearest Neighbors`:** Classification by proximity to the nearest neighbors <br> **`SVM`:** Classifier that seeks the best boundary between classes <br> **`MLP (Neural Network)`:** Classifier based on a neural network with one hidden layer and 50 neurons, **`hidden_layer_sizes=(50,)`:** Neural network with 1 hidden layer of 50 neurons, **`random_state=42`:** Ensures that the neural network always gives the same result. | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/5.DefineModels.png" width="4000"/>|
Create a **`for`** loop that iterates over each model in the model in the models dictionary <br> **`model.fit(x_train,y_train)`:** Train each model. <br> **`model.score(x_test, y_test)`:** Evaluate the model and return the accuracy <br> Save the results in the dictionary results <br>Print the acurracy with 4 decimal places.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/6.TrainEvaluateModels.png" width="4000"/>|
Plot a bar chart with the accuracies of each model. <br> Label the axes and put the title "Comparison of Algorithms (MNIST 8x8)". <br> **`plt.show*()`:** Show the graph| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.3TrainingMultipleAlgorithms/Images/7.VisualizeResults.png" width="4000"/>|