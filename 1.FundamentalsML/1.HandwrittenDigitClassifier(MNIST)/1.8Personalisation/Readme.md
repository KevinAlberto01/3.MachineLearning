<p align = "center" >
    <h1 align = "Center"> Data Argumentation </h1>
</p>

<p align = "center" >
    <h2 align = "Center">🎯 Objetives 🎯</h2>
</p>

To facilitate the visualization of the information generated by the project, an interactive dashboard was created using **Streamlit**.  This dashboard allows the user to explore and better understand the results in an intuitive and user-friendly manner.

<p align = "center" >
    <h2 align = "Center">📑 Structure 📑</h2>
</p>

As part of the dashboard implementation process, I first created a layout draft to better visualize the desired data presentation. The layout consists of two rows: the first row contains three columns, while the second row contains four columns. This structure was designed to improve the organization and readability of the dashboard's content.

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/DraftDashboard.png" width="2000"/>

<p align = "center" >
    <h2 align = "Center">📝 Results 📝 </h2>
</p>

<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/Dashboard.png" width="2000"/>

<p align = "center" >
    <h2 align = "Center"> 💻 Program explication 💻 </h2>
</p> 

|Pseudocode| Image of the program|
|----------|---------------------|
**Streamlit (st)**: Used to create the web interface. <br> **Pandas (pd):** To handle and display tabular data. <br> **NumPy (np):** For numerical operations. <br> **Matplotlib (plt)** and **Seaborn (sns)**: For creating visualizations. <br> **io and PIL.Image:** To handle and display images in Streamlit. <br> **Scikit-learn:** To work with the SVM model, split the data and evaluate the model's performance.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/1.ImportLibraries.png" width="4000"/>|
Set the page layout to be wider.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/2.ConfigureStramlitPage.png" width="4000"/>|
It is used to remove the top margin and stick the page title to the top.|<img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/3.StyleCSSPage.png" width="4000"/>|
This is the main title of the page, which is centered and has a large size.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/4.PageTitle.png" width="4000"/>|
The digits dataset from sklearn is loaded. <br> **`x`** contains the 8x8 pixel images (in the form of 64-feature vectors), and **`y`** contains the labels (digits from 0 to 9).| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/5.CargePreparationDataset.png" width="4000"/>|
GridSearchCV is used to search for the best parameters for the SVM model (**`C`** and **`kernel`**), using 3-fold cross-validation. <br> Predictions are made on the test set using the model with the best parameters found. <br> The confusion matrix, the classification report (with precision, recall, f1-score), and the overall accuracy of the model are calculated.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/6.TrainingModelWithGridsearch.png" width="4000"/>|
This function creates a donut chart to visualize the model's accuracy in an appealing way.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/7.CreateDonutChartAccuracy.png" width="4000"/>|
The interface is created in multiple columns to visually organize the content: <br> Divide the screen into three columns, where the central column (**`col2_top`**) takes up the most space.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/8.StructureColumns2.png" width="4000"/>|
Column 1: Dataset and Accuracy <br> Show the information of the dataset (**`x`** and **`y`**), and the model's accuracy in a donut chart. <br> Column 2: True Images vs Predictions <br> Show the first 10 images from the test set along with their true labels and predictions. <br> Column 3: Confusion matrix <br> Visualize the confusion matrix in heatmap format.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/9.ShowInformationColumns2.png" width="4000"/>|
Additional information is displayed in columns for details on class distribution, the best parameters, the normalized data, and the classification report.| <img src = "https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.8Personalisation/Images/10.ShowOtherDetails.png" width="4000"/>|