<img src="https://images7.alphacoders.com/737/thumb-1920-737400.jpg">

# <span style="color:cyan"> Proyecto ADL: Estimación de ratings en anime :star2:
### <span style="color:lightblue"> Descripción del Proyecto :clipboard:
Este proyecto utiliza el conjunto de datos provisto por Academia Desafío Latam. El objetivo es usar Machine Learning para crear un sistema de ayude a estimar el puntaje que una serie/película debería tener dadas sus características.

### <span style="color:lightblue"> Evaluación :chart_with_upwards_trend:
En este caso al ser un problema de regresión, las métricas consideradas son R2, MAE y MAPE. Estas se utilizan para evaluar la precisión y la capacidad de explicación del modelo, así como para cuantificar el error absoluto y porcentual en las predicciones.

### <span style="color:orange"> Herramientas Utilizadas :wrench:
- Python 3.9.17
- Bibliotecas de análisis de datos: Pandas, NumPy.
- Bibliotecas de visualización: Matplotlib, Seaborn.
- Biblioteca de aprendizaje automático: scikit-learn.

### <span style="color:orange"> Estructura del Proyecto :open_file_folder:
- anime.csv: Archivo CSV que contiene los datos.
- proyecto.ipynb: Un Jupyter notebook que contiene el código Python para el análisis de datos.
- funciones.py: Archivo Python que contiene las funciones utilizadas para este proyecto.

### <span style="color:orange"> Cómo usar este proyecto :question:
1. Coloca el archivo anime.csv en la misma carpeta que este proyecto.
2. Abre el Jupyter notebook proyecto.ipynb y ejecuta las celdas de código paso a paso para explorar y analizar los datos.

### <span style="color:orange"> Contenido del Jupyter notebook :page_facing_up:
El Jupyter notebook proporciona un análisis completo de los datos, que incluye:
- Exploración de datos: Resumen estadístico, visualización de datos, identificación de valores nulos, etc.
- Preprocesamiento de datos: Limpieza de datos, manejo de valores faltantes, codificación de variables categóricas, etc.
- Análisis de características: Visualización de relaciones entre las variables.
- Modelado y predicción: Entrenamiento de modelos de aprendizaje automático para predecir el puntaje de cada serie/película.
- Evaluación del modelo: Evaluación de las métricas R2, MAE y MAPE accuracy.
