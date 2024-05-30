Desarrollo de nuevos productos alimenticios funcionales con ingredientes endémicos de México, utilizando inteligencia artificial (IA)
==============================

Esta investigación explora el uso de técnicas de IA, específicamente 
Aprendizaje Automático (ML) y Algoritmos Genéticos, para impulsar la innovación
práctica en la ingeniería de alimentos. El enfoque innovador integra diversas 
fuentes de datos sobre características relevantes de los alimentos, como 
moléculas y perfiles de sabor con sus respectivos grupos funcionales de sabor, 
así como valores nutricionales. Se desarrolló un modelo de clasificación 
binaria de Bosque Aleatorio que compara productos alimenticios por pares, 
aprendiendo a identificar similitudes con un rendimiento prometedor (AUC PR 
de 0.898). Este modelo se integró en un algoritmo genético iterativo que 
propuso listas de ingredientes optimizadas para replicar productos objetivo 
como queso, leche y mantequilla. Los candidatos generados lograron puntajes de
similitud entre 0.5 y 0.7, indicando una probabilidad del 80% de conservar 
propiedades nutricionales y sensoriales comparables a los productos originales,
pero con una composición de ingredientes completamente diferente. Esta 
metodología demuestra el potencial de la IA para innovar en el diseño y 
personalización de productos alimenticios, contribuyendo a la diversificación, 
sostenibilidad y accesibilidad en la industria alimentaria.

¿Cómo ejecutar el código correctamente?
--------
1. Crea un nuevo ambiente virtual (puedes usar el comando `python -m venv venv`) y activalo (`source venv/bin/activate`).
2. Instala las dependencias correspondientes: `pip install -r requirements.txt`. 
3. Ejecuta con python el siguiente script the python `general_pipeline.py` python script. Este script contiene el orden correcto de ejecución para descargar los datos, procesarlos y finalmente crear un modelo.

Importante: Para la ejecución total del flujo se requiere una API key de Edamam, la cual se coloca en un archivo `.env`.  

Organización del proyecto:
------------
    ├── README.md            <- The top-level README for developers using this project.
    ├── data
    │   ├── interim          <- Intermediate data that has been transformed.
    │   ├── processed        <- The final, canonical data sets for modeling.
    │   └── raw              <- The original, immutable data dump.
    │
    ├── models               <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks            <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                           the creator's initials, and a short `-` delimited description, e.g.
    │                           `1-DHM-initial_data_exploration`.
    │
    ├── notebooks_discovery  <- Additional Jupyter notebooks that helped to develop the branch.
    │
    ├── references           <- Data references to the project, manuals, and all other explanatory materials.
    │
    ├── reports              <- Generated analysis of the model as HTML, PDF, LaTeX, etc.
    │   └── figures          <- Generated graphics and figures.
    │
    ├── requirements.txt     <- The requirements file for reproducing the analysis environment.
    │
    ├── setup.py             <- Makes project pip installable (pip install -e .) so src can be imported
    │
    ├── src                  <- Source code for use in this project.
    │   ├── __init__.py      <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download the data.
    │   │
    │   ├── demo           <- Scripts to execute the demo.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling.
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │                     predictions.
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │
    │
    ├── general_pipeline.py  <- Executable that follows the correct order of scripts to replicate the results. 
    │
    ├── ReleaseNotes.md  <- Notes that complement information on the file. 
    │
    ├── .gitignore         <- Git file to ignore files and documents.
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

