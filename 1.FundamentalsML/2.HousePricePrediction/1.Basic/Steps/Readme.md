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

<h3 align="center">2.1 Procesamiento de Datos</h3>

<h4 align="center">2.1.1 Carga de datos</h4>
En el primer paso del proyecto, se cargó el conjunto de datos Ames Housing, obteniendo un total de 2,930 filas y 82 columnas.
Esto proporciona una base rica y detallada de características que describen las propiedades, incluyendo aspectos como tamaño, calidad, ubicación y más.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/1.1.1.jpeg?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />

>[!NOTE]
>Esta etapa es crucial para tener una visión general del dataset, identificar posibles errores o datos faltantes, y planificar los siguientes pasos de limpieza y análisis.


<h4 align="center">2.1.2 Verificar los null values</h4>

Verificamos los valores nulos presentes en el dataset. En esta etapa, solo se realiza una inspección visual para entender cómo está compuesta nuestra base de datos, sin tomar aún decisiones sobre qué variables eliminar o conservar. Esto nos permite tener una mejor idea de la calidad y completitud de los datos antes de continuar con el análisis.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/1.2.jpeg?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />


<h4 align="center">2.1.3 Identificación de datos duplicados y análisis de tipos de datos</h4>

En este paso combinamos dos tareas importantes: primero, detectamos los datos duplicados para garantizar la calidad del dataset; luego, analizamos los tipos de datos presentes. Esta última acción es fundamental para planificar futuros procesos, ya que cada variable puede requerir un tratamiento diferente según su tipo.

| **Valores duplicados** | **Tipos de datos** |
|------------------------|--------------------|
| <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/1.4.jpeg?raw=true" alt="Valores duplicados" style="width: 90%; height: auto;"> | <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/1.3.jpeg?raw=true" alt="Tipos de datos" style="width: 90%; height: auto;"> |


<h3 align="center">2.2 Exploratory Data Analysis</h3>
<h4 align="center">2.2 Verificar los null values</h4>

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
