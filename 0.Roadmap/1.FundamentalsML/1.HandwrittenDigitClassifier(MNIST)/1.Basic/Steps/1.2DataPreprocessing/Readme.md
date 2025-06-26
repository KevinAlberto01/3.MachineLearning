<p align = "center" >
    <h1 align = "Center"> 2.Data Preprocessing</h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

Transform the raw data into a format that is more suitable for the model to learn patterns more quickly and accurately.

* **1.Normalization / Standardization**
    - Scale the pixel values to be in a more uniform range
        - Mean 0 and variance 1 with StandardScaler, or between 0 and 1 by dividing by 255
    - This makes the training more stable and faster, and improves the performance of some models.

* **2.Conversion to vectors**
    - Since each image is 8x8, it is flattened into a vector of 64 values so that the model can work with it. 
    - Many traditional machine learning models (logistic regression, SVM, KNN) expect the input data to be vectors (not matrices or images).


<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

View the original images to confirm that the data you are using is reasonable and makes sense.

| <p align = "left"> It's just a way to check that you are working with real images of handwritten digits. <br> It doesn't matter that the data is internally normalized, the images always look like digits because imshow() knows how to interpret the matrices as images.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.2DataPreprocessing/Images/Results.png" width="4000"/>|
|----------------------------|----------------------------------------------------------------------------|

Check the normalized numerical values to ensure that the preprocessing was done correctly.

| <p align = "left"> **Negative values** represent pixels that are below average. <br> <br> Positive values represent pixels that are above the average. <br> <br> Values close to 0 represent pixels with an average value.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.2DataPreprocessing/Images/DataNormalized.png" width="500"/>|
|----------------------------|----------------------------------------------------------------------------|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**NumPy:** Used to work with arrays and matrices, and it has functions for mathematics and statistics <br> **Pandas:** Used to manipulate and analyze data, particularly with DataFrames (tables of rows and columns).<br> **matplotlib.pyplot:** Used to create graphs, and pyplot is specifically for generating different types of plots (e.g., bar, line, scatter).<br> **load_digits from sklearn.datasets:** Used to load a dataset of handwritten digits, which is commonly used in classification projects.<br> **sklearn.preproccessing.StandardScaler:** Normalize the data | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.2DataPreprocessing/Images/1.ImportLibraries.png" width="4000"/>|
Load the digits dataset that comes with scikit-learn, it contains images of handwritten digits of 8x8 pixels. <br> **`digits.data`:** it contains the images, but each image is in the form of a vector of 64 elements (8x8). <br> **`digits.target`:** Contains the labels of the digits (0-9). | <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.2DataPreprocessing/Images/2.LoadData.png" width="4000"/>|
Create a figure with 1 row and 5 columns to display 5 images. <br> Take the first 5 images (`x[0]`, `x[1]`, ...) <br> **`x[i]`:** is converted from a vector to an 8x8 matrix with `reshape(8, 8)`. <br> **`cmap=gray`:** the images are displayed in grayscale.<br> **`ax.set_title`:** Hide the axes <br> **`plt.show()`:** Show the images|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.2DataPreprocessing/Images/3.ShowImages.png" width="4000"/>|
Normalize the data using StandardScaler() from sklearn,it makes the data have a mean of 0 and a standard deviation of 1, it is important for Machine Learning algorithms to work better. <br> **`scaler.fit_transform()`:** Calculates the mean and standard deviation of x and transforms the data <br> **`x_scaled`:** Now it contains the normalized data.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.2DataPreprocessing/Images/4.NormalizationStandard.png" width="4000"/>|
Print the first 5 rows of the normalized data.<br>Each row still represents an 8x8 image, but the values are now normalized floating-point numbers (mean 0, standard deviation 1).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.2DataPreprocessing/Images/5.ShowFirstRows.png" width="4000"/>|
