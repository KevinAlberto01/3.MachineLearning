<h1 align="center"  style="margin-bottom: -10px;">🏠 House Price Prediction with Machine Learning 🏠</h1>
<div align="center">

🌐 Este README está disponible en: [Inglés](Readme.md) | [Español](ReadmeESP.md) 🌐

</div>

<h2 id="table-of-contents" align="center">📑 Table of Contents</h2>

1. [Descripción](#descripcion)
2. [2.Carpetas dentro de "1.Basic"](#basic)
   - [2.1.Local](#local)
3. [Nivel 2 – Modelo Avanzado](#nivel2)
4. [Nivel 3 – Despliegue del Modelo](#nivel3)
5. [Objetivos](#objetivos2)


<h2 id="descripcion" align="center">📜 Descripción 📜</h2>

En esta carpeta se encuentran tres subcarpetas principales: 

- Local
- Steps 
- streamlit

A continuación, se explicará cómo ejecutar el proyecto y el motivo por el cual se organizó de esta manera.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

>[!NOTE]
>Esta imagen es importante porque cada carpeta está diseñada para un uso distinto. Aunque contienen los mismos programas, se ejecutan de manera diferente según el contexto.

<h2 id="basic" align="center">2.Carpetas dentro de "1.Basic"</h2>

<h3 id="local" align="center">2.1.Local</h3>

La carpeta está diseñada para ejecutar el proyecto de manera local, permitiendo aplicar toda la lógica del modelo, realizar mejoras, y visualizar los resultados a través de un dashboard interactivo con Streamlit. Aquí puedes probar el flujo completo desde el preprocesamiento hasta la visualización final, sin necesidad de conexión a internet o despliegue externo.

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.Basic/Local
python 1.LOGICA.py
```
>[!NOTE]
>Esta sección está enfocada únicamente en la lógica y exportación de los modelos. No permite realizar predicciones, ya que esa funcionalidad se encuentra en el segundo programa que se presenta a continuación.

```bash
git clone https://github.com/KevinAlberto01/3.MachineLearning.git
cd 3.MachineLearning/1.FundamentalsML/2.HousePricePrediction/1.Basic/Local
streamlit run 2.DASHBOARD.py
```
>[!NOTE]
>Esta sección está destinada a visualizar la predicción generada con la lógica del programa anterior. Si deseas analizar en detalle cómo construí esa lógica, te recomiendo ir a la carpeta "Steps" o importar directamente el paso anterior, donde explico cómo integré todo el flujo de trabajo.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.2Local.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

<h3 align="center">2.2.Steps</h3>

Esta carpeta está diseñada para explicar en detalle la lógica de desarrollo seguida en el proyecto. Aquí se documentan tanto los pasos del flujo de Machine Learning como los procesos adicionales, incluyendo el diseño y estructura del dashboard interactivo creado con Streamlit.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.3Steps.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

>[!NOTE]
>Todo está documentado dentro del README, pero únicamente como explicación de los pasos y de la lógica utilizada.
Si deseas ejecutar el programa, es necesario subir los archivos y correrlo de forma local.

<h3 align="center">2.3.streamlit</h3>

Esta carpeta está diseñada para ejecutar el programa directamente en Streamlit.io, lo que permite visualizar el dashboard de forma interactiva a través de un enlace público. Esta opción es ideal para compartir una demostración del proyecto sin necesidad de instalar nada localmente.

<img src="https://github.com/KevinAlberto01/3.MachineLearning/blob/main/0.Roadmap/1.FundamentalsML/2.HousePricePrediction/1.Basic/Img/1.4streamlit.png?raw=true" alt="Tipos de datos" style="width: 100%; height: auto;"> 

>[!NOTE]
>En esta carpeta no es posible ejecutar el programa localmente directamente. Para visualizar el dashboard, es necesario subir el proyecto a Streamlit.io, donde la plataforma se encargará de ejecutarlo en línea.
Aun así, la estructura del proyecto ya está preparada para facilitar esa carga, por lo que no tendrás problemas para mostrar correctamente el dashboard una vez desplegado.



