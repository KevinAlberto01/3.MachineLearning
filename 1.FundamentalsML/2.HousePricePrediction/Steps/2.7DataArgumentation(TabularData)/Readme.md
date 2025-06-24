<p align = "center" >
    <h1 align = "Center"> Data Argumentation (Tabular Data)</h1>
</p>

This program implements and compares different Data Augmentation methods applied to a dataset of house prices. Its main objective is to analyze how the generation of synthetic data affects the performance of a Linear Regression model in terms of Mean Squared Error (MSE).

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

* Evaluate the impact of different Data Augmentation techniques on a regression problem.
* Compare the Mean Squared Error (MSE) between the original data set and the augmented sets.
* Identify whether any augmentation techniques improve model performance or introduce unnecessary noise.
* Visualize the results clearly using a comparison plot.
* Ensure modular and reusable code for future testing with other regression models.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Final Comparation </h4>
</p>

The MSE (Mean Squared Error) values shown in the final comparison indicate the mean squared error of each model with its respective Data Augmentation technique. However, there is something strange about these extremely high values.


|Pseudocode| Image of the program|
|----------|---------------------|
|original:  <br> This is the error of the model trained on the unmodified data. <br> But this number is too large, suggesting a problem in the scale of the data. <br> <br>jittering: <br> This MSE is much more reasonable and is within an expected range. <br>It appears that the model with jittering is performing better than the others.<br><br>knn: <br> This number is absurdly large, indicating a problem with the KNN synthetic data generation. <br> Possible errors: <br> The generated values are out of scale. <br> The output (y) values do not match well with the newly generated points. <br><br> generative: <br> Again, this number is too high. <br> There is probably a problem with the addition of noise in augment_generative.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/Images/FinalComparation.png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Graphical Data </h4>
</p>

|Pseudocode| Image of the program|
|----------|---------------------|
|| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/Images/GraphicalData.png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|**pandas** / **numpy:** Data manipulation. <br> **matplotlib.pyplot:** Results visualization. <br> /**sklearn.model_selection:** Division into training and testing. <br> **LinearRegression** / **mean_squared_error:** Model and evaluation metrics. <br> **StandardScaler:** Feature scaling. <br> **NearestNeighbors:** For Data Augmentation based on KNN. <br> **warnings:** To ignore certain warning messages.
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/Images/1.ImportLibraries.png" width="4000"/>|
|Load the dataset **AmesHousing_cleaned.csv.** <br> Separate predictor variables (**X**) and target variable (**y →   `saleprice`**). <br> Convert categorical variables to one-hot encoding (**`pd.get_dummies`**). <br> Standardize features using **`StandardScaler()`**. <br> Split the data into **training (80%) and test (20%).**| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/Images/2.LoadingPreparingDataSet.png" width="4000"/>|
|Do not make changes to the data. <br> It is used as a base reference. <br> Add Gaussian (**normal**) noise to numerical features. <br> Parameter **`sigma=0.01`:** controls the noise intensity. <br> Duplicate the dataset with modified versions of the original samples. <br> Create synthetic data by interpolating values between a sample and its **nearest neighbors (KNN).** <br> A random neighbor is selected and a new sample is generated with linear interpolation. <br> **Apply Gaussian** noise to the target variable y instead of the **X** features. <br> **`sigma=0.05`:** adjust the amount of noise in the house price.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/Images/3.DataAugmentation.png" width="4000"/>|
|Train a Linear Regression model and evaluate its MSE.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/Images/4.TrainingEvaluation.png" width="4000"/>|
|Evaluate the four Data Augmentation methods. <br> Capture errors and store np.nan if a method fails.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/Images/5.EvaluatingAugmentation.png" width="4000"/>|
|Create a bar chart **comparing** the **MSE** of each method. <br> Show the **MSE values** for each bar. <br> Indicate errors in red.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/Images/6.Visualization.png" width="4000"/>|