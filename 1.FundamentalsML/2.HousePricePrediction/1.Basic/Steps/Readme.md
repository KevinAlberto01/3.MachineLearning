<h1 align="center"  style="margin-bottom: -10px;">🏠 House Price Prediction with Machine Learning 🏠</h1>
<div align="center">

🌐 This README is available in: [English](README.md) | [Spanish](README.es.md)

</div>

<h2 align="center">📑 Table of Contents</h2>

1. [Introduction](#1-introduction)
2. [Development Process](#2-development-process)
   - [1.1 Data Processing](#21-data-processing)
   - [1.2 Exploratory Data Analysis (EDA)](#22-exploratory-data-analysis-eda)
   - [1.3 Training Multiple Algorithms](#23-training-multiple-algorithms)
   - [1.4 Evaluation Metrics](#24-evaluation-metrics)
   - [1.5 Optimization (Tuning & Hyperparameters)](#25-optimization-tuning--hyperparameters)
   - [1.6 Manual Feature Engineering](#26-manual-feature-engineering)
   - [1.7 Data Augmentation (Tabular)](#27-data-augmentation-tabular)
3. [Final Results](#3-final-results)
4. [Technologies Used](#4-technologies-used)
5. [How to Run](#5-how-to-run)


<h2 align="center">1. 📌 Introduction </h2>

Este es un proyecto básico de regresión en el que se implementa una pipeline completa de Machine Learning (excepto la etapa de data augmentation) para predecir el precio de una vivienda. El modelo toma en cuenta características clave como la calidad, el tamaño y la ubicación, utilizando datos históricos del mercado inmobiliario.

A continuación, se explicará cada paso para mostrar la lógica con la que se resolvió el problema.

<h2 align="center">2.Development Process</h2>

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td style="width: 50%; vertical-align: top; text-align: left; padding-right: 20px;">
      El proceso de desarrollo de este proyecto sigue una estructura lógica en etapas que permiten construir un modelo de regresión robusto y comprensible. <br>
      Comenzamos con la carga y limpieza de los datos, seguido de un análisis exploratorio para entender mejor las variables y su impacto en el precio. <br>
      Luego, se entrenan múltiples algoritmos para comparar su rendimiento utilizando métricas adecuadas. <br>
      Para la evaluación, utilizamos estas métricas con el fin de comparar y seleccionar el mejor algoritmo. <br>
      Después, optimizamos el modelo elegido para obtener mejores resultados y predicciones. <br>
      Posteriormente, integramos todo el flujo en un único programa robusto. <br>
      Finalmente, realizamos predicciones utilizando el modelo entrenado y presentamos los resultados en un dashboard interactivo.
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/spanish.png?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />
    </td>
  </tr>
</table>



<div align="center">
  <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/spanish.png?raw=true" alt="Dashboard Preview" width="600"/>
</div>

<h3 align="center">2.1 Procesamiento de Datos</h3>

<h3 align="center">2.2 Exploratory Data Analysis</h3>

<h3 align="center">2.3 Training Multiple Algorithms</h3>

<h3 align="center">2.4 Evaluation Metrics</h3>

<h3 align="center">2.5 Optimization (Tuning & Hyperparameters)</h3>

<h3 align="center">2.6 Feature Engineering Manual</h3>

<h3 align="center">2.7 Data Argumentation (Tabular Data)</h3>

<h2 align="center">3.Final Results</h2>
<h2 align="center">4.Technologies Used</h2>
<h2 align="center">▶️ How to Run</h2>

```bash
git clone https://github.com/yourusername/housing-ml-project.git
cd housing-ml-project
pip install -r requirements.txt
streamlit run streamlit_app.py
