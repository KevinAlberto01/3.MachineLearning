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

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/1.3.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">



<h3 align="center">2.2 Exploratory Data Analysis</h3>

<h4 align="center">2.2.1 Heatmap</h4>
Primero, realizamos un heatmap de toda la base de datos para visualizar qué variables presentan relaciones entre sí. Sin embargo, debido a la gran cantidad de datos, el gráfico no permite una visualización clara. Por ello, en el siguiente paso filtramos y enfocamos el análisis en las variables más relevantes para obtener una mejor interpretación.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.1.1.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

Después, se imprimen las variables más relevantes que se utilizarán para generar un heatmap reducido. El objetivo es obtener una mejor visualización y comprensión de las relaciones entre las variables más influyentes en este caso específico.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.1.2.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

Por último, utilizamos las variables más relevantes para generar un heatmap reducido, en el cual se observa que las dos primeras variables presentan la mayor correlación (mayor intensidad de color). Esta observación es útil, ya que en los siguientes pasos estas variables pueden ser modificadas o utilizadas para mejorar el modelo.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.1.3.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

<h4 align="center">2.2.2 Pairplot</h4>

En este paso se utiliza un pairplot para visualizar el comportamiento de las variables seleccionadas. Ya que hemos identificado dos variables con alta correlación, es importante observar su distribución y relación visualmente para entender mejor cómo interactúan entre sí.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.1.4.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

<h4 align="center">2.2.3 Estadísticas descriptivas</h4>

En base a las variables seleccionadas (Gr Liv Area y Overall Qual), obtenemos información más detallada sobre sus relaciones con el resto del dataset. Sin embargo, es importante recordar que SalePrice es nuestra variable objetivo (y), ya que nuestro propósito principal es predecir su comportamiento.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.1.5.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">

<h4 align="center">2.2.4 Histogramas</h4>

Este paso nos ayuda a visualizar la distribución de los datos de cada variable. A través de los histogramas podemos:

- Detectar la forma de la distribución (normal, sesgada, etc.).
- Identificar posibles sesgos y valores atípicos (outliers).
- Evaluar si alguna variable requiere una transformación (como logaritmos o escalado).
- Observar la distribución general entre variables numéricas.
- Tomar decisiones sobre el tipo de preprocesamiento que podría mejorar el rendimiento del modelo.

Esta visualización es clave para entender mejor nuestros datos antes de construir modelos predictivos.

|Distribución de Gr Liv Area|Distribución de SalePrice|Distribution of Overall Quality|
|------------------------|------------------------|-------------------| 
|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.2.1.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.2.2.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.2.3.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> |

>[!IMPORTANT]
>A continuación veremos el comportamiento y las combinaciones de cada una en caso de seleccionarlo, pero aun no lo escogemos

|SalePrice and Gr Liv Area|SalePrice and Overall Qual|Sale Price, Gr Liv Area, overall qual |
|------------------------|------------------------|-------------------| 
|Ambas variables presentan una distribución sesgada y contienen valores atípicos (outliers) extremos. Por ello, es recomendable utilizar MinMaxScaler, ya que es más seguro para mantener la escala original sin verse afectado por dichos valores atípicos. <br> Por otro lado, también se podría usar StandardScaler, pero hay que considerar que los outliers pueden influir significativamente en la media y la desviación estándar, afectando el escalado.| SalePrice es una variable continua, mientras que Overall Qual es una variable ordinal con valores de 1 a 10 que representan la calidad. <br> No es necesario normalizar Overall Qual, ya que es un número discreto con significado específico. Por lo tanto, se podría aplicar escalado únicamente a SalePrice usando MinMaxScaler o StandardScaler.| MinMaxScaler es una opción segura si queremos que todas las variables estén en el rango [0, 1]. <br> StandardScaler puede ser útil si Gr Liv Area y SalePrice siguen una distribución normal. <br> Por otro lado, Overall Qual es una variable ordinal, por lo que es recomendable mantenerla sin escalar para conservar su significado.|

<h4 align="center">2.2.5 Boxplot</h4>
Un boxplot es una representación gráfica que muestra la distribución de una variable numérica, destacando su mediana, cuartiles y posibles valores atípicos (outliers).<br>
Permite identificar la dispersión, simetría y la presencia de datos extremos de forma rápida y visual.<br><br>

<h5 align="center">2.2.5.1 Boxplot SalePrice</h5>
Observamos varios puntos por encima del rango intercuartílico, lo que indica la presencia de valores atípicos elevados. Esto sugiere que existen viviendas con precios considerablemente más altos que el promedio del dataset.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.3.1.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"><br>

<h5 align="center">2.2.5.2 Boxplot Gr Liv Area</h5>

El boxplot de Gr Liv Area también muestra la presencia de outliers elevados, con los bigotes superiores extendiéndose más allá que en otros casos, lo que indica que algunas viviendas tienen áreas habitables significativamente mayores al rango típico.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.3.2.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"><br>

<h5 align="center">2.2.5.3 Boxplot OverallQual</h5>

El boxplot de OverallQual presenta valores concentrados principalmente por debajo de la mediana, con pocos o ningún outlier visible, lo que refleja que la mayoría de las viviendas tienen una calidad general dentro de un rango más limitado.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.3.3.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> <br>

Finalmente, analizamos el valor p (p-value) para evaluar la significancia estadística de nuestras variables con respecto a la variable objetivo (SalePrice).

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.3.4.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h4 align="center">2.2.6 Distribución de los datos</h4>

Necesitamos calcular el skewness (coeficiente de sesgo) de cada variable para cuantificar el sesgo en su distribución, lo cual es importante para decidir qué acciones tomar en el preprocesamiento.
Después de calcular el skewness, observamos que las variables Gr Liv Area y Overall Qual presentan un sesgo positivo, es decir, la mayoría de los valores son bajos, pero existen algunos valores muy altos que desvían la distribución.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.3.5.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

Después, filtramos las filas donde alguna variable tenga un valor menor o igual a 0, ya que en datos reales ese tipo de valores no deberían existir (por ejemplo, una superficie o precio negativo no tiene sentido). Esta verificación nos ayuda a detectar errores o inconsistencias en los datos antes de entrenar el modelo.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.3.6.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

Por último, pero no menos importante, volvemos a verificar los valores nulos, pero esta vez solo en las variables que realmente nos interesan para el modelo. Esto nos permite enfocar el preprocesamiento en las columnas relevantes y tomar decisiones informadas sobre cómo manejar los datos faltantes

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.3.7.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h4 align="center">2.2.7 Aplicamos logaritmos</h4>
Como pudimos observar, los datos presentan distribuciones sesgadas, lo cual puede afectar el rendimiento de los modelos de regresión.
Para corregir este problema, aplicamos una transformación logarítmica, lo que nos permite reducir el sesgo y acercar la distribución a una forma más simétrica. A continuación, se muestra una comparación antes y después de aplicar el logaritmo:

| Antes del logaritmo | Después del logaritmo |
|----------------------------------------------|--------------|
|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.4.1.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;">|<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.4.2.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> |

<h4 align="center">2.2.8 Normalizamos los datos</h4>
Antes de finalizar el análisis de distribución, es importante considerar que la variabilidad entre escalas también puede afectar el rendimiento del modelo.
Por ello, es necesario normalizar los datos para que las variables tengan valores en rangos similares. Esto ayuda a que el modelo aprenda de manera más eficiente y equitativa.
En este paso, observamos los valores de las variables seleccionadas para decidir qué tipo de escalado aplicar.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.4.1.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

Por último, realizamos una gráfica para verificar si, tras aplicar la normalización y corregir el sesgo, la distribución de los datos se ha ajustado adecuadamente.
Esta visualización nos permite confirmar si la variable se aproxima a una distribución normal (forma de campana), lo cual es deseable para muchos algoritmos de Machine Learning.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/2.4.2.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 


<h3 align="center">2.3 Training Multiple Algorithms</h3>

Pasamos a la etapa de entrenamiento del modelo, donde aplicaremos diferentes algoritmos de regresión para comparar su rendimiento.
Dado que se trata de un problema de regresión, se entrenarán los siguientes modelos:

- KNN Regressor
- SVR (Support Vector Regressor)
- Redes Neuronales (MLP Regressor)
- Ridge Regression (Regularización L2)
- Lasso Regression (Regularización L1)
- LightGBM
- XGBoost

El objetivo de este paso es evaluar qué modelo se adapta mejor a los datos en función de las métricas de desempeño, interpretabilidad y complejidad computacional.

<h4 align="center">2.3.1 KNN Regressor</h4>

KNN Regressor predice el valor de un punto tomando el promedio de los k vecinos más cercanos según una métrica de distancia. Es un modelo basado en instancias, sin entrenamiento real, ideal cuando se espera que datos similares tengan valores de salida similares.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.1KNNRegressor.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h4 align="center">2.3.2 SVR(Support Vector Regressor)</h4>

SVR es una extensión del algoritmo de Support Vector Machine para tareas de regresión. Busca ajustar una línea (o hiperplano) que prediga los datos con un margen de tolerancia definido, minimizando los errores fuera de ese margen. Es útil para datos no lineales y ofrece buena generalización.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.2SVR.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h4 align="center">2.3.3 Redes Neuronales (MLP)</h4>

El MLP Regressor (Perceptrón Multicapa) es una red neuronal con una o más capas ocultas que aprende patrones complejos mediante propagación hacia adelante y retropropagación. Es ideal para capturar relaciones no lineales entre las variables y se adapta bien a conjuntos de datos con múltiples características.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.3MLPRegressor.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h4 align="center">2.3.4 LightGBM</h4>
LightGBM es un algoritmo de Gradient Boosting optimizado para velocidad y eficiencia. Construye árboles de decisión de forma hoja a hoja (leaf-wise) en lugar de nivel por nivel, lo que mejora el rendimiento y precisión. Es ideal para grandes volúmenes de datos y tareas de regresión con alta dimensionalidad.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.4LightGBM.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 
<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.4LightGBM2.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h4 align="center">2.3.5 Ridge Regression (L2 Regularization)</h4>
Ridge Regression es una extensión de la regresión lineal que aplica regularización L2 para reducir el sobreajuste. Penaliza los coeficientes grandes al agregar su suma cuadrada al término de pérdida, lo que estabiliza el modelo cuando hay multicolinealidad o muchas variables.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.5RidgeRegression.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h4 align="center">2.3.6 Lasso Regression (L1 Regularization)</h4>
Lasso Regression utiliza regularización L1, que penaliza la suma absoluta de los coeficientes. Esto no solo reduce el sobreajuste, sino que también puede eliminar variables irrelevantes, ya que tiende a llevar algunos coeficientes exactamente a cero, funcionando como una forma de selección de variables.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.6LassoREgressor.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h4 align="center">2.3.7 XGBoost</h4>

XGBoost (Extreme Gradient Boosting) es un algoritmo de gradient boosting altamente optimizado y eficiente. Utiliza técnicas avanzadas como regularización, poda de árboles, y paralelización para mejorar tanto la precisión como el rendimiento computacional. Es muy popular en competencias de Machine Learning por su capacidad de manejar datos complejos y ruidosos.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/1.FundamentalsML/2.HousePricePrediction/1.Basic/Steps/Img/3.Training/3.1.7XGBoost.jpeg?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 


<h3 align="center">2.4 Evaluation Metrics</h3>

<h3 align="center">2.5 Optimization (Tuning & Hyperparameters)</h3>

<h2 align="center">3.Final Results</h2>
<h2 align="center">4.Technologies Used</h2>
<h2 align="center">▶️ How to Run</h2>

```bash
git clone https://github.com/yourusername/housing-ml-project.git
cd housing-ml-project
pip install -r requirements.txt
streamlit run streamlit_app.py
