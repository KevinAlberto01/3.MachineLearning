<h1 align="center"  style="margin-bottom: -10px;">1.Fundamentals ML</h1>
<div align="center"> </div>

<div align="center">

🌐 Este README está disponible en: [Inglés](Readme.md) | [Español](ReadmeESP.md) 🌐

</div>

<h2 id="table-of-contents" align="center">📑 Table of Contents</h2>

1. [Descripción](#descripcion)
2. [1.Clasificador de Dígitos Escritos a Mano (MNIST)](#MNIST)
    - [Objetivos](#objetivos1)
    - [Enfoque](#enfoque1)
    - [Resultados](#Resultados1)
3. [2.Predicción de Precios de Casas](#casas)
   - [Objetivos](#objetivos2)
    - [Enfoque](#enfoque2)
    - [Resultados](#Resultados2)
4. [3.Análisis de Sentimiento (Reseñas de Películas)](#peliculas)
   - [Objetivos](#objetivos3)
    - [Enfoque](#enfoque3)
    - [Resultados](#Resultados3)




<h2 id="descripcion" align="center">📜 Descripción 📜</h2>

Esta sección está diseñada para desarrollar una comprensión sólida de los conceptos fundamentales en Machine Learning. Abarca desde la manipulación y preparación de datos hasta la implementación y evaluación básica de modelos.
Los proyectos incluidos permiten aplicar estos conceptos de manera práctica para sentar las bases de un flujo de trabajo completo en ciencia de datos.

Los proyectos incluidos en esta etapa son:

- **Clasificador de Dígitos Escritos a Mano (MNIST)**
    - Un modelo de clasificación para reconocer números escritos a mano utilizando técnicas de aprendizaje supervisado.

- **Predicción de Precios de Casas**
    - Un modelo de regresión que estima el valor de propiedades en función de sus características estructurales y de ubicación.

- **Análisis de Sentimientos (Reseñas de Películas)**
    - Un modelo de clasificación que identifica el sentimiento (positivo o negativo) en reseñas de películas mediante procesamiento de lenguaje natural.

Cada uno de estos proyectos está diseñado para fortalecer habilidades esenciales en el desarrollo de modelos de Machine Learning, interpretación de resultados y construcción de flujos de trabajo reproducibles.

>[!NOTE]
>Cada uno enfocado en diferentes aspectos del aprendizaje automático.


<h2 id="MNIST" align="center">1.Clasificador de Dígitos Escritos a Mano (MNIST)</h2>


<div align="center"> </div>


<h4 id="objetivos1" align="center">🎯 Objetivos 🎯</h2>

Desarrollar un modelo de clasificación que reconozca dígitos escritos a mano utilizando el dataset load_digits de la librería scikit-learn.
El objetivo es implementar una pipeline completa de Machine Learning, desde la carga y preprocesamiento de datos hasta el entrenamiento, optimización y despliegue del modelo para resolver un problema de clasificación de imágenes.

<h4 id="enfoque1" align="center">🔎 Enfoque 🔎</h2>

Clasificación de imágenes en escala de grises de 8x8 píxeles que representan dígitos del 0 al 9.

- Exploración y visualización para entender la distribución de clases y los patrones de los píxeles.
- Preprocesamiento y normalización de datos para preparar el modelo.
- Entrenamiento y comparación de varios algoritmos: Regresión Logística, KNN, SVM y MLP.
- Optimización de hiperparámetros para mejorar la precisión del modelo.
- Desarrollo de un dashboard interactivo con Streamlit para realizar predicciones en tiempo real y visualizar resultados.

<h4 id="resultados1" align="center">✅ Resultados ✅ </h2>

Se logró una alta precisión en todos los modelos; se seleccionó SVM como modelo final tras la optimización por su equilibrio entre desempeño y eficiencia.

Se desarrolló un dashboard funcional con Streamlit que permite al usuario visualizar predicciones y explorar la lógica de clasificación.

|Basic Model|Advance model|Deployment|
|------------------------|------------------------|-------------------| 
|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/Img/Dashboard.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/Img/3.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/Img/3.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">|



<h2 id="casas" align="center">2.Predicción de Precios de Casas</h2>

<h4 id="objetivos2" align="center">🎯 Objetivos 🎯</h2>

Predecir el valor de propiedades residenciales en función de múltiples características como el número de habitaciones, calidad, tamaño, ubicación y año de construcción.  
El objetivo es construir un modelo confiable que generalice bien a nuevos datos.

<h4 id="enfoque2" align="center">🔎 Enfoque 🔎</h2>

Implementación de algoritmos de regresión (Regresión Lineal, Árbol de Decisión, Random Forest, KNN y LightGBM).  
Se aplicaron técnicas de Ingeniería de Características y Optimización de Hiperparámetros para mejorar el rendimiento y la interpretabilidad del modelo.

<h4 id="resultados2" align="center">✅ Resultados ✅ </h2>

Se logró un modelo predictivo robusto con un R² alto y un RMSE bajo en los datos de prueba.  
El proyecto se desarrolló en tres etapas: una versión básica con modelos iniciales, una versión avanzada con ingeniería de características y optimización, y finalmente una etapa de despliegue.  
En esta última, se creó un dashboard interactivo con Streamlit que permite ingresar nuevos datos y visualizar las predicciones en tiempo real.

|Basic Model|Advance model|Deployment|
|------------------------|------------------------|-------------------| 
|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/Img/1.BasicModel.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/Img/2.AdvanceModel.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/Img/2.AdvanceModel.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">|

<h2 id="peliculas" align="center">3.Análisis de Sentimiento (Reseñas de Películas)</h2>

<h4 id="objetivos3" align="center">🎯 Objetivos 🎯</h2>
Analyze movie reviews to classify sentiment (positive or negative).

<h4 id="enfoque3" align="center">🔎 Enfoque 🔎</h2>

Natural Language Processing (NLP) and text analysis techniques to extract semantic patterns.

<h4 id="resultados3" align="center">✅ Resultados ✅ </h2>

<img src = "" width="2000"/>