<h1 align="center"  style="margin-bottom: -10px;">🏠 House Price Prediction with Machine Learning 🏠</h1>
<div align="center">

🌐 Este README está disponible en: [Inglés](Readme.md) | [Español](ReadmeESP.md) 🌐

</div>

<h2 align="center">📑 Table of Contents</h2>

1. [Descripción](#descripcion)
2. [Proceso de Desarrollo](#desarollo)
   - [1.1.Procesamiento de Datos](#datos)
      - [1.1.1.Carga de datos](#Cdatos)
      - [1.1.2.Explorando las dimensiones del conjunto de datos](#Nvalues)
      - [1.1.3 Número de ejemplos por clase](#Dduplicados)
   - [1.2.Análisis Exploratorio de Datos (EDA)](#eda)
      - [1.2.1.Distribución de clases](#heatmap)
      - [1.2.2.Ejemplos de las clases](#pairplot)
      - [1.2.3.Normalizamos los datos](#desc)
    - [1.3 Entrenamiento de Múltiples Algoritmos](#Malgoritmos)
    - [1.4 Métricas de Evaluación](#metricas)
    - [1.5 Optimización (Ajuste y Hiperparámetros)](#optimizacion)
    - [1.6 Agrupar (1/2)](#agrupar1)
    - [1.7 Personalización](#perso1)
    - [1.8 Join all (2/2)](#join2)
4. [Resultados Finales](#resultados)
5. [Tecnologías Utilizadas](#tech)
6. [Cómo Ejecutar](#ejecutar)

<h2 id="descripcion" align="center">📜 Descripción 📜</h2>

Este es un proyecto de clasificación en el que se implementa una pipeline completa de Machine Learning (excepto la etapa de data augmentation) para clasificar dígitos escritos a mano. Cabe destacar que se trabaja con imágenes, por lo que el modelo debe predecir el valor numérico representado visualmente.

A continuación, se explican cada uno de los pasos seguidos para resolver el problema, detallando la lógica aplicada en cada etapa del proceso.

<h2 id="desarollo" align="center">Proceso de Desarrollo</h2>

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td style="width: 50%; vertical-align: top; text-align: left; padding-right: 20px;">
    El desarrollo de este proyecto sigue una estructura lógica por etapas que permite construir un modelo de clasificación robusto y comprensible. <br>
    Comenzamos con la carga y preparación de los datos, seguidos de un análisis exploratorio para entender mejor la distribución de clases y el contenido visual del dataset. <br>
    Luego, se entrenan múltiples algoritmos de clasificación con el fin de comparar su rendimiento utilizando métricas adecuadas.
    En la etapa de evaluación, utilizamos estas métricas para seleccionar el modelo con mejor desempeño.
    Posteriormente, optimizamos los modelos para mejorar sus resultados y predicciones. <br>
    Una vez finalizado este proceso, integramos todo el flujo de trabajo en un programa unificado y funcional.
    Finalmente, realizamos pruebas utilizando el modelo entrenado y visualizamos los resultados en un dashboard.
    </td>
    <td style="width: 50%;">
      <img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/Spanish.png?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />
    </td>
  </tr>
</table>

<h3 id="desarollo" align="center">1.1 Procesamiento de Datos</h3>


<h4 id="Cdatos" align="center">1.1.1 Carga de datos</h4>

En este primer paso, cargamos el conjunto de datos que utilizaremos a lo largo del proyecto. Específicamente, empleamos el dataset load_digits, el cual es proporcionado por la librería sklearn.datasets.
Este dataset es ampliamente utilizado como punto de partida para tareas de clasificación de imágenes debido a su tamaño manejable y su formato estandarizado.

En esta etapa, también separamos los datos en dos variables principales:

**X**: que contiene las representaciones numéricas (flattened) de las imágenes.
**y**: que contiene las etiquetas correspondientes a cada imagen, indicando el dígito que representa (del 0 al 9).

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/1.LoadData/1.0.png?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />

>[!NOTE]
>Este dataset nos permite entrenar y evaluar modelos de forma rápida sin necesidad de preprocesamiento complejo o recursos computacionales intensivos.


<h4 id="Nvalues" align="center">1.1.2 Explorando las dimensiones del conjunto de datos</h4>

A continuación, verificamos cuántas muestras tenemos disponibles en las variables X e y, asegurándonos de que ambas tengan la misma cantidad de elementos, lo cual es fundamental para el entrenamiento del modelo.

La variable X representa los datos de entrada, es decir, los valores de intensidad de los píxeles de cada imagen. Cada imagen ha sido transformada en un vector unidimensional de 64 elementos (8x8), lo que permite que los algoritmos de Machine Learning puedan procesarlas como datos numéricos.

Por otro lado, la variable y contiene las etiquetas o valores objetivo, que indican el número que representa cada imagen (del 0 al 9). Estas etiquetas servirán como referencia durante el entrenamiento, permitiendo al modelo aprender a reconocer y clasificar los patrones visuales que caracterizan a cada dígito.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/1.LoadData/1.1.png?raw=true" alt="Dashboard Preview" style="width: 100%; height: auto;" />


<h4 id="Dduplicados" align="center">1.1.3 Número de ejemplos por clase </h4>


Finalmente, analizamos las clases presentes en el conjunto de datos, así como la cantidad de ejemplos disponibles para cada una de ellas. Este análisis nos permite entender la distribución general del dataset y detectar posibles desequilibrios entre clases, lo cual es un aspecto crucial en los problemas de clasificación.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/1.LoadData/1.2.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

<h3 id="eda" align="center">1.2 Exploratory Data Analysis (EDA)</h3>

<h4 id="heatmap" align="center">1.2.1 Distribución de clases</h4>

Como primer paso en el análisis exploratorio, es importante visualizar gráficamente la distribución de los datos para identificar cuántos ejemplos existen por clase. Esta representación nos permite observar de forma clara si alguna clase está desbalanceada, es decir, si hay dígitos que aparecen con mayor o menor frecuencia en comparación con otros.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/2.DataPreprocessing/2.1.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">


<h4 id="pairplot" align="center">1.2.2 Ejemplos de las clases</h4>

A continuación, observaremos algunos ejemplos visuales de cada una de las clases presentes en el dataset. El objetivo de esta visualización es comprender cómo se representan gráficamente los dígitos y qué tipo de variaciones pueden existir dentro de una misma clase.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/2.DataPreprocessing/2.2.png?raw=true?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

Esto nos permite tener una mejor idea de los patrones que el modelo deberá aprender a detectar y clasificar, considerando que los datos escritos a mano pueden presentar diferencias en forma, tamaño y estilo según cada muestra.

<h4 id="desc" align="center">1.2.3.Normalizamos los datos</h4>

Observamos que la distribución de clases en el conjunto de datos es bastante uniforme, es decir, el número de ejemplos por clase es similar. Gracias a este balance, no es necesario aplicar técnicas adicionales de balanceo o muestreo.

Por lo tanto, el único paso previo al entrenamiento de los modelos que consideramos necesario en esta etapa es la normalización de los datos. Este proceso ajusta la escala de los valores de entrada, facilitando el aprendizaje de los algoritmos y mejorando su rendimiento y estabilidad.


<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/2.DataPreprocessing/2.3.png?raw=true?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

<h3 id="Malgoritmos" align="center">1.3.Training Multiple Algorithms</h3>

Pasamos a la etapa de entrenamiento del modelo, donde aplicaremos diferentes algoritmos de clasificación para comparar su rendimiento.
Dado que se trata de un problema de clasificación, se entrenarán los siguientes modelos:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)

El objetivo de este paso es evaluar cuál modelo se adapta mejor a los datos, considerando métricas de desempeño, interpretabilidad y complejidad computacional.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/3.training/3.2.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

<h3 id="metricas" align="center">1.4.Métricas de Evaluación</h3>

Tras evaluar el rendimiento de todos los modelos, observamos algo curioso: todos alcanzaron el mismo nivel de precisión (accuracy), lo que sugiere que cualquiera de ellos podría funcionar correctamente. Por lo tanto, aún no se ha seleccionado un modelo definitivo.

<br>
<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/4.EvaluationM/4.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h3 id="optimizacion" align="center">1.5.Optimization (Tuning & Hyperparameters)</h3>

Al optimizar todos los modelos, me percaté de que dos de ellos mostraron una mejora significativa en su desempeño: Support Vector Machine (SVM) y Multi-layer Perceptron (MLP). Ambos modelos lograron superar a los demás en términos de precisión y estabilidad durante las pruebas. Sin embargo, considerando factores como la complejidad computacional, la interpretabilidad y el tiempo de entrenamiento, optaremos por utilizar el modelo SVM como el modelo final para este proyecto.
 
<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/5.Optimization/5.6.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h3 id="optimizacion" align="center">1.6.Agrupar (1/2)</h3>

En este paso, nos enfocamos en presentar visualmente los elementos más relevantes del programa, eliminando aquellos modelos, métricas y componentes que ya no son necesarios para simplificar el análisis.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/6.JoinAll/5.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h3 id="perso1" align="center">1.7 Personalización</h3>
Para el desarrollo del dashboard, utilicé la librería Streamlit y elaboré un boceto inicial con el propósito de definir la estructura visual y funcional del proyecto antes de proceder con su implementación.

<h3 id="join2" align="center">1.7 Personalización</h3>
<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/7Personalization/Dashboard.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 


<h2 id="resultados" align="center">4.4.Resultados Finales</h2>

📂 Dataset utilizado: `load_digits`, proporcionado por la librería [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) *(disponible en sklearn.datasets)*
🧠 Tipo de aprendizaje: Supervisado
📈 Tipo de problema: Clasificación
⚙️ Algoritmo principal: SVM
🧪 Nivel del modelo: Básico
💻 Lenguaje utilizado: Python
👤 Tipo de proyecto: Personal / Portafolio

<h2 id="tech" align="center">5.Tecnologías Utilizadas</h2>

📊 Manipulación y análisis de datos

- pandas
- numpy

📈 Visualización

- matplotlib
- seaborn
- pillow

🤖 Machine Learning

- scikit-learn

📦 Desarrollo de aplicaciones
- streamlit

<h2 id="ejecutar" align="center">6.Como Ejecutar el programa</h2>

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Steps/img/Program.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/0.Roadmap/1.FundamentalsML/1.HandwrittenDigitClassifier(MNIST)/1.Basic/Local/
1.Basic/Local
python 0.LOGICA.py
```

>[!IMPORTANT]
>Estos comandos se utilizan exclusivamente para ejecutar la lógica del programa, incluyendo el flujo de trabajo de Machine Learning.

