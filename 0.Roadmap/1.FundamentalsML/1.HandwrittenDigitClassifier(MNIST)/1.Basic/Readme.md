<h1 align="center" style="margin-bottom: -10px;">🏠 House Price Prediction with Machine Learning 🏠</h1>
<div align="center">

🌐 This README is available in: [English](Readme.md) | [Spanish](ReadmeESP.md) 🌐

</div>

<h2 id="table-of-contents" align="center">📑 Table of Contents</h2>

1. [Description](#description)  
2. [2. Folders inside "1.Basic"](#basic)
   - [2.1. Local](#local)
   - [2.2. Steps](#steps)
   - [2.3. Streamlit](#streamlit)

<h2 id="description" align="center">📜 Description 📜</h2>

This directory contains three main subfolders:

- Local  
- Steps  
- Streamlit  

Below is an explanation of how to run the project and the reasoning behind this structure.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.png?raw=true" alt="Data types" style="width: 100%; height: auto;">

>[!NOTE]
>This image is important because each folder is designed for a specific purpose. Although they contain the same programs, they are executed differently depending on the context.

<h2 id="basic" align="center">2. Folders inside "1.Basic"</h2>

<h3 id="local" align="center">2.1. Local</h3>

This folder is intended for running the project locally, allowing you to apply the entire model logic, perform improvements, and view the results through a Streamlit dashboard.  
Here, you can test the full pipeline—from preprocessing to final visualization—without requiring an internet connection or external deployment.

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Local
python 1.LOGICA.py

```
>[!NOTE]
>This section focuses solely on the logic and export of the models. It does not allow predictions, as that functionality is included in the second program shown below.

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Local
streamlit run 1.HandWrittenDigitClassifier(MNIST).py
```

>[!NOTE]
>This section is designed to display predictions generated using the logic from the previous program.
If you want a deeper understanding of how that logic was built, we recommend going to the "Steps" folder or importing the previous script, where the full pipeline is explained step-by-step.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Img/local.png?raw=true" alt="Local view" style="width: 100%; height: auto;"> <h3 id="steps" align="center">2.2. Steps</h3>
This folder is intended to explain in detail the development logic followed in the project. It documents both the steps of the Machine Learning pipeline and additional processes, including the design and structure of the interactive dashboard built with Streamlit.

>[!NOTE]
>Since this was my first Machine Learning project, I chose to include each step in the code with explanations to make the workflow easier to understand.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Img/steps.png?raw=true" alt="Steps" style="width: 100%; height: auto;">

>[!IMPORTANT]
>Everything is documented inside the README, but only as an explanation of the steps and logic used.
If you want to run the program, you’ll need to upload the files and execute them locally.

<h3 id="streamlit" align="center">2.3. Streamlit</h3>
This folder is designed to run the program directly on Streamlit.io, allowing you to visualize the dashboard via a public link.
This option is ideal for sharing a project demonstration without needing to install anything locally.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Img/streamlit.png?raw=true" alt="Streamlit view" style="width: 100%; height: auto;">

>[!NOTE]
>This folder cannot be executed locally as-is.
To visualize the dashboard, you must upload the project to Streamlit.io, where the platform will run it online.
Nevertheless, the project structure is already prepared to make this upload seamless, so the dashboard should display correctly once deployed.