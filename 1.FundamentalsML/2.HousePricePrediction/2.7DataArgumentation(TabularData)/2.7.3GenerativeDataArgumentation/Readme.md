<p align = "center" >
    <h1 align = "Center"> Jittering (Optional)</h1>
</p>

Applies Data Augmentation on a housing price data set by adding Gaussian noise to the numeric variables. It then saves the augmented data set to a CSV file and displays a comparison between the original and modified values.

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

Create an augmented version of the original data set by adding a small amount of Gaussian noise to the numerical variables. This can help improve the robustness of Machine Learning models by making them less sensitive to minor variations in the data.

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

|Pseudocode| Image of the program|
|----------|---------------------|
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/Data.png" width="4000"/>|

|Pseudocode| Image of the program|
|----------|---------------------|
|d|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/Information.png" width="4000"/>|


<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Jitters) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J1.ImportLibraries.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J2.LoadData.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J3.StaticalSummary.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J4.NumericalVariables.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J5.ScaleData.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J6.ApplyJittering.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J7.RestoreValues.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J8.HandlingCategorical.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J9.CombineNumericalCategorical.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J10.AugmentedDataset.png" width="4000"/>|
||<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/J11.FinalMessages.png" width="4000"/>|
<p align = "center" >
    <h1 align = "Center"> Generative Data Argumentation </h1>
</p>


<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<p align = "center" >
    <h4 align = "Center"> Data </h4>
</p>

|Data (without Jittering)| Data (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/ComparationData.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/ComparationData(Jittering).png" width="4000"/>|

<p align = "center" >
    <h4 align = "Center"> Graph </h4>
</p>

|Graph (without Jittering)| Graph (With Jittering)|
|----------|---------------------|
| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/ComparationGraph.png" width="4000"/>| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/2.7DataArgumentation(TabularData)/2.7.3GenerativeDataArgumentation/Images/ComparationGraph(Jittering).png" width="4000"/>|

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication (Jittering (Perturbation Leve)) 💻</h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
|| <img src = "" width="4000"/>|