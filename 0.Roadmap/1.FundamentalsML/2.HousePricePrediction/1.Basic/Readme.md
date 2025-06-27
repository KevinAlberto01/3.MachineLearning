<h1 align="center"  style="margin-bottom: -10px;">🏠 House Price Prediction with Machine Learning 🏠</h1>
<div align="center">
🌐 This README is available in: [English](Readme.md) | [Spanish](ReadmeESP.md) 🌐
</div>

<h2 id="table-of-contents" align="center">📑 Table of Contents</h2>

1. [Descripción](#descripcion)
2. [2.Carpetas dentro de "1.Basic"](#basic)
   - [2.1.Local](#local)
   - [2.2.Steps](#steps)
   - [2.3.Streamlit](#streamlit)

<h2 id="description" align="center">📜 Description 📜</h2>

This folder contains three main subfolders:

- Local  
- Steps  
- Streamlit

Below, you will find how to run the project and the reason why it was organized this way.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.png?raw=true" alt="Data Types" style="width: 100%; height: auto;"> 

>[!NOTE]  
>This image is important because each folder is designed for a different purpose. Although they contain the same programs, they are executed differently depending on the context.

<h2 id="basic" align="center">2. Folders inside "1.Basic"</h2>

<h3 id="local" align="center">2.1. Local</h3>

The folder is designed to run the project locally, allowing you to apply all the model logic, make improvements, and visualize the results through an interactive dashboard with Streamlit.  
Here, you can test the full workflow from preprocessing to final visualization, without the need for internet connection or external deployment.

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.Basic/Local
python 1.LOGICA.py
```

>[!NOTE]
>This section is focused only on the logic and model export. It does not allow making predictions, as that functionality is found in the second program shown below.


```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.Basic/Local
streamlit run 2.DASHBOARD.py
```

>[!NOTE]
>This section is intended to visualize the prediction made with the logic from the previous script.
If you want to analyze my logic, it’s important to go to the “Steps” folder or import the previous step — that’s where I explain how I integrated everything.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.2Local.png?raw=true" alt="Data Types" style="width: 100%; height: auto;"> 

<h3 id="steps" align="center">2.2.Steps</h3>

This folder is designed to explain in detail the development logic followed in the project.
Here, both the steps of the Machine Learning pipeline and additional processes are documented, including the design and structure of the interactive dashboard created with Streamlit.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.3Steps.png?raw=true" alt="Data Types" style="width: 100%; height: auto;">

>[!NOTE]
>Everything is inside the README, but it’s only an explanation of the steps and logic.
If you want to run the program, you need to upload it and run it locally.

<h3 id="streamlit" align="center">2.3.streamlit</h3>

This folder is designed to run the program directly on Streamlit.io, allowing you to view the dashboard interactively through a public link.
This option is ideal for sharing a demo of the project without needing to install anything locally.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.4streamlit.png?raw=true" alt="Data Types" style="width: 100%; height: auto;">

>[!NOTE]
>You cannot run this folder locally. To view the dashboard, you need to upload the project to Streamlit.io, where the platform will execute it online.
Still, the project structure is already prepared for easy deployment, so you won’t have any trouble displaying the dashboard once it’s published.