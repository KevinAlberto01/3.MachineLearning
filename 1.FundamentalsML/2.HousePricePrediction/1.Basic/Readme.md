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

En esta carpeta se encuentran tres subcarpetas principales: Local, Steps y streamlit.
A continuación, se explicará cómo ejecutar el proyecto y el motivo por el cual se organizó de esta manera.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h2 align="center">2.Carpetas dentro de "1.Basic"</h2>

<h3 align="center">2.1.Local</h3>

La carpeta está diseñada para ejecutar el proyecto de manera local, permitiendo aplicar toda la lógica del modelo, realizar mejoras, y visualizar los resultados a través de un dashboard interactivo con Streamlit. Aquí puedes probar el flujo completo desde el preprocesamiento hasta la visualización final, sin necesidad de conexión a internet o despliegue externo.

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.Basic/Local
python 1.LOGICA.py
```

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.Basic/Local
streamlit run 2.DASHBOARD.py
```

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.2Local.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h3 align="center">2.2.Steps</h3>

Esta carpeta está diseñada para explicar en detalle la lógica de desarrollo seguida en el proyecto. Aquí se documentan tanto los pasos del flujo de Machine Learning como los procesos adicionales, incluyendo el diseño y estructura del dashboard interactivo creado con Streamlit.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.3Steps.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 


<h3 align="center">2.3.streamlit</h3>

Esta carpeta está diseñada para ejecutar el programa directamente en Streamlit.io, lo que permite visualizar el dashboard de forma interactiva a través de un enlace público. Esta opción es ideal para compartir una demostración del proyecto sin necesidad de instalar nada localmente.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.4streamlit.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

>[!NOTE]
>En esta carpeta no podemos ejecutarlo localmente sino que tenemos que entrar a la pagina de streamli.io para poder subirlo y que la pagina lo ejecute pero te dejo la estructura con eso no tendras problemas para que se muestre tu dashboard

