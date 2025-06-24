<h1 align="center"  style="margin-bottom: -10px;">🏠 House Price Prediction with Machine Learning 🏠</h1>
<div align="center">

🌐 This README is available in: [English](README.md) | [Spanish](README.es.md)

</div>

<h2 align="center">📑 Table of Contents</h2>

1. [Introduction](#1-introduction)
2. [Development Process](#2-development-process)
   - [2.1 Data Processing](#21-data-processing)
   - [2.2 Exploratory Data Analysis (EDA)](#22-exploratory-data-analysis-eda)
   - [2.3 Training Multiple Algorithms](#23-training-multiple-algorithms)
   - [2.4 Evaluation Metrics](#24-evaluation-metrics)
   - [2.5 Optimization (Tuning & Hyperparameters)](#25-optimization-tuning--hyperparameters)
   - [2.6 Manual Feature Engineering](#26-manual-feature-engineering)
   - [2.7 Data Augmentation (Tabular)](#27-data-augmentation-tabular)
3. [Final Results](#3-final-results)
4. [Technologies Used](#4-technologies-used)
5. [How to Run](#5-how-to-run)


<h2 align="center">1. 📌 Introduction </h2>

Este es un proyecto básico de regresión en el que se implementa una pipeline completa de Machine Learning (excepto la etapa de data augmentation) para predecir el precio de una vivienda. El modelo toma en cuenta características clave como la calidad, el tamaño y la ubicación, utilizando datos históricos del mercado inmobiliario. Los resultados y análisis se presentan de forma interactiva en un dashboard intuitivo y visual.

A continuación, se explicará cada paso para mostrar la lógica con la que se resolvió el problema.

<div align="center">
  <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/spanish.png?raw=true" alt="Dashboard Preview" width="600"/>
</div>


## 5. ▶️ How to Run

```bash
git clone https://github.com/yourusername/housing-ml-project.git
cd housing-ml-project
pip install -r requirements.txt
streamlit run streamlit_app.py
