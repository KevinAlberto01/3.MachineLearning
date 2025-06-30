<h1 align="center" style="margin-bottom: -10px;">🏠 House Price Prediction with Machine Learning 🏠</h1>
<div align="center"></div>

<div align="center">
🌐 This README is available in: [English](Readme.md) | [Spanish](ReadmeESP.md) 🌐
</div>

<h2 id="table-of-contents" align="center">📑 Table of Contents</h2>

1. [Description](#description)  
2. [Level 1 – Basic Model](#level1)  
3. [Level 2 – Advanced Model](#level2)  
4. [Level 3 – Model Deployment](#level3)  
5. [Objectives](#objectives2)

<h2 id="description" align="center">📜 Description 📜</h2>

This is a supervised learning project focused on classifying handwritten digits using the `load_digits` dataset from the `sklearn.datasets` library.  
The dataset consists of 8x8 grayscale images representing digits from 0 to 9, which are converted into numerical vectors for processing.

<h2 id="level1" align="center">Level 1 – Basic Model</h2>

This first version focuses on loading and performing preliminary analysis of the dataset, as well as training simple models to establish a prediction baseline.  
A basic dashboard was also developed using Streamlit to enable interactive visualization of the data and initial results, allowing users to explore variable behavior and better understand the dataset.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/Img/Dashboard.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;">

<h2 id="level2" align="center">Level 2 – Advanced Model</h2>

This version incorporates more sophisticated techniques, such as Feature Engineering and Data Augmentation, to improve dataset quality.  
Hyperparameter tuning and a thorough comparison between advanced models are also performed to select the most accurate and generalizable one.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/Img/4.png?raw=true" alt="Data types" style="width: 100%; height: auto;">

<h2 id="level3" align="center">Level 3 – Model Deployment</h2>

This stage focuses on building a robust interactive dashboard using Streamlit. Unlike the previous stages, this version allows users to:

- Enter new data manually or upload files.
- Get real-time predictions.
- Visualize charts and metrics in a clear and intuitive way.

It is also ready to be connected to the cloud, making it accessible from any device without requiring local code execution.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/Img/4.png?raw=true" alt="Data types" style="width: 100%; height: auto;">

<h2 id="objectives2" align="center">🎯 Objectives 🎯</h2>

- Implement a complete Machine Learning pipeline to solve an image classification problem, from data loading to final prediction.  
- Explore and understand the `load_digits` dataset from `scikit-learn`, identifying structure, class distribution, and visual characteristics.  
- Apply multiple classification algorithms (Logistic Regression, KNN, SVM, and MLP) to compare performance and understand their pros and cons.  
- Evaluate models using performance metrics such as accuracy to select the most suitable model.  
- Optimize selected models through hyperparameter tuning to improve precision and stability.  
- Develop an interactive dashboard using Streamlit to visually present model functionality, from prediction to result interpretation.

<p align="center">
    <h2 align="center">💻​ Versions 💻​</h2>
</p>

| Basic Model | Advanced Model | Deployment |
|------------------------|------------------------|-------------------|  
| <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/Img/Dashboard.jpeg?raw=true" alt="Data types" style="width: 100%; height: auto;"> | <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/Img/4.png?raw=true?raw=true" alt="Data types" style="width: 100%; height: auto;"> | <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/Img/4.png?raw=true?raw=true" alt="Data types" style="width: 100%; height: auto;"> |
