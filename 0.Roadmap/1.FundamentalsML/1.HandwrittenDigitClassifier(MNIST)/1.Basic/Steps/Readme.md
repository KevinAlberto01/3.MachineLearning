<h1 align="center" style="margin-bottom: -10px;">🏠 Handwritten Digit Classification with Machine Learning 🏠</h1>
<div align="center">

🌐 This README is available in: [English](Readme.md) | [Spanish](ReadmeESP.md) 🌐

</div>

<h2 align="center">📑 Table of Contents</h2>

1. [Description](#description)  
2. [Development Process](#development)  
   - [1.1. Data Processing](#data)  
      - [1.1.1. Data Loading](#Cdata)  
      - [1.1.2. Exploring Dataset Dimensions](#Nvalues)  
      - [1.1.3. Number of Examples per Class](#Dduplicated)  
   - [1.2. Exploratory Data Analysis (EDA)](#eda)  
      - [1.2.1. Class Distribution](#heatmap)  
      - [1.2.2. Class Examples](#pairplot)  
      - [1.2.3. Data Normalization](#desc)  
   - [1.3. Training Multiple Algorithms](#Malgorithms)  
   - [1.4. Evaluation Metrics](#metrics)  
   - [1.5. Optimization (Tuning & Hyperparameters)](#optimization)  
   - [1.6. Join All (1/2)](#join1)  
   - [1.7. Personalization](#perso1)  
   - [1.8. Final Integration (2/2)](#join2)  
4. [Final Results](#results)  
5. [Technologies Used](#tech)  
6. [How to Run](#run)

<h2 id="description" align="center">📜 Description 📜</h2>

This is a classification project where a complete Machine Learning pipeline is implemented (excluding the data augmentation stage) to classify handwritten digits. It is worth noting that the data consists of images, so the model must predict the numerical value each image represents.

Each step taken to solve the problem is explained below, outlining the logic applied throughout the process.

<h2 id="development" align="center">Development Process</h2>

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td style="width: 50%; vertical-align: top; text-align: left; padding-right: 20px;">
      The development of this project follows a logical step-by-step structure that allows building a robust and understandable classification model.  
      We start with data loading and preparation, followed by exploratory analysis to understand the class distribution and the dataset's visual content.  
      Then, multiple classification algorithms are trained in order to compare their performance using suitable evaluation metrics.  
      During the evaluation stage, these metrics help select the best-performing model.  
      Later, all models are optimized to improve their performance and predictions.  
      Once the process is complete, the entire workflow is integrated into a unified and functional program.  
      Finally, we test the trained model and visualize the results through a dashboard.
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/Spanish.png?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />
    </td>
  </tr>
</table>

<h3 id="data" align="center">1.1 Data Processing</h3>

<h4 id="Cdata" align="center">1.1.1 Data Loading</h4>

In this first step, we load the dataset that will be used throughout the project. Specifically, we use the `load_digits` dataset, which is provided by the `sklearn.datasets` module.  
This dataset is widely used as an entry point for image classification tasks due to its manageable size and standardized format.

In this stage, we also split the data into two main variables:

**X**: containing the flattened numerical representations of the images.  
**y**: containing the corresponding labels for each image, indicating the digit it represents (from 0 to 9).

<h4 id="Nvalues" align="center">1.1.2 Exploring Dataset Dimensions</h4>

Next, we check how many samples are available in both X and y, ensuring they match in quantity — which is essential for training.  
X contains the input data, that is, the pixel intensity values of each image, flattened into 64-element (8x8) vectors.  
y contains the labels that indicate which digit each image represents, which the model will learn to predict.

<h4 id="Dduplicated" align="center">1.1.3 Number of Examples per Class</h4>

Finally, we analyze the classes present in the dataset and the number of samples available for each.  
This helps us understand the dataset distribution and identify potential class imbalances — a key aspect of classification problems.

<h3 id="eda" align="center">1.2 Exploratory Data Analysis (EDA)</h3>

<h4 id="heatmap" align="center">1.2.1 Class Distribution</h4>

As the first step in exploratory analysis, we visualize the data distribution to determine how many samples exist per class.  
This allows us to clearly identify if any class is overrepresented or underrepresented compared to others.

<h4 id="pairplot" align="center">1.2.2 Class Examples</h4>

Next, we visualize examples from each class to understand how digits are represented and how they vary within the same class.  
This helps us get a clearer sense of the patterns the model must learn to classify, taking into account variations in shape, size, and handwriting style.

<h4 id="desc" align="center">1.2.3 Data Normalization</h4>

We observed that the class distribution is fairly balanced, meaning that each digit has a similar number of samples.  
Therefore, no additional balancing techniques are needed.  
The only preprocessing step applied before training is data normalization, which adjusts the scale of input values to improve the learning performance of the models.

<h3 id="Malgorithms" align="center">1.3 Training Multiple Algorithms</h3>

We now proceed to the model training stage, where we apply several classification algorithms to compare their performance.  
Since this is a classification problem, the following models are trained:

- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Multi-Layer Perceptron (MLP)

The goal is to evaluate which model fits the data best, considering performance, interpretability, and computational cost.

<h3 id="metrics" align="center">1.4 Evaluation Metrics</h3>

After evaluating all models, something interesting happened: they all achieved the same level of accuracy.  
This suggests that any of them could perform well, and therefore no final model has been selected yet.

<h3 id="optimization" align="center">1.5 Optimization (Tuning & Hyperparameters)</h3>

After tuning all models, two of them showed significant performance improvements: SVM and Multi-layer Perceptron (MLP).  
Both outperformed the others in terms of accuracy and stability.  
Considering factors like computational cost, interpretability, and training time, we chose SVM as the final model for this project.

<h3 id="join1" align="center">1.6 Join All (1/2)</h3>

In this step, we focus on visually presenting the most important elements of the program by removing models, metrics, and components that are no longer necessary.

<h3 id="perso1" align="center">1.7 Personalization</h3>

For the dashboard development, I used the Streamlit library and created an initial mockup to define the visual and functional structure of the project before implementation.

<h3 id="join2" align="center">1.8 Final Integration (2/2)</h3>

In this step, we performed a complete test of the dashboard to ensure everything was aligned properly and functioning as expected.  
The interface displays predicted values clearly and cleanly.

<h2 id="results" align="center">3. Final Results</h2>

📂 Dataset used: `load_digits`, provided by [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) *(available in sklearn.datasets)*  
🧠 Learning type: Supervised  
📈 Problem type: Classification  
⚙️ Main algorithm: SVM  
🧪 Model level: Basic  
💻 Language: Python  
👤 Project type: Personal / Portfolio

<h2 id="tech" align="center">4. Technologies Used</h2>

📊 **Data Manipulation & Analysis**

- pandas  
- numpy  

📈 **Visualization**

- matplotlib  
- seaborn  
- pillow  

🤖 **Machine Learning**

- scikit-learn  

📦 **App Development**

- streamlit  

<h2 id="run" align="center">5. How to Run the Program</h2>

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Local/
python 0.LOGICA.py
```

>[!IMPORTANT]
>These commands are intended to execute the core logic of the program, including the full Machine Learning workflow.