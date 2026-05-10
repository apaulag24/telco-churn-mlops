# Sistema de Clasificación y Despliegue MLOps de Modelos de Ensamble para el Abandono en Telecomunicaciones

# Descripción general del proyecto

Este proyecto documenta y ejecuta el ciclo de vida completo de un modelo de Machine Learning (MLOps) diseñado para predecir la probabilidad de abandono (`churn`) en una compañía de telecomunicaciones. Toda la documentación técnica, los hallazgos del análisis exploratorio y los detalles de la arquitectura se encuentran disponibles y renderizados para su consulta en la página oficial del proyecto: [TU_ENLACE_AQUÍ].

A diferencia de un ejercicio puramente estadístico, este repositorio abarca desde la caracterización del comportamiento del usuario y la mitigación de la fuga de información, hasta la serialización y puesta en producción del artefacto predictivo. El flujo de trabajo integra la construcción de Pipelines robustos, el empaquetado en entornos aislados mediante `Docker` y una validación automatizada a través de `GitHub Actions`, garantizando que el sistema pase de la fase de experimentación local a una arquitectura escalable y técnicamente íntegra.

# Fuente de datos

El conjunto de datos utilizado es el Telco Customer Churn Dataset, que contiene información sobre una compañía de telecomunicaciones que proporciona servicios de telefonía e Internet a **7,043 clientes en California**. El dataset permite analizar el comportamiento de los usuarios para identificar perfiles con alta propensión al abandono.

**Diccionario de variables:**
El conjunto consta de atributos demográficos, servicios contratados e información financiera:

* **gender:** Género del cliente (`Male`, `Female`).  
* **SeniorCitizen:** Cliente adulto mayor (`1`: Sí, `0`: No).  
* **Partner:** Cliente con pareja (`1`: Sí, `0`: No).  
* **Dependents:** Cliente con dependientes (`1`: Sí, `0`: No).  
* **tenure:** Meses de antigüedad del cliente (0–72).  
* **PhoneService:** Tiene servicio telefónico (`1`: Sí, `0`: No).  
* **MultipleLines:** Tiene múltiples líneas (`Yes`, `No`, `No phone service`).  
* **InternetService:** Tipo de servicio de internet (`DSL`, `Fiber optic`, `No`).  
* **OnlineSecurity:** Tiene seguridad en línea (`Yes`, `No`, `No internet service`).  
* **OnlineBackup:** Tiene respaldo en línea (`Yes`, `No`, `No internet service`).  
* **DeviceProtection:** Tiene protección de dispositivos (`Yes`, `No`, `No internet service`).  
* **TechSupport:** Tiene soporte técnico (`Yes`, `No`, `No internet service`).  
* **StreamingTV:** Tiene TV en streaming (`Yes`, `No`, `No internet service`).  
* **StreamingMovies:** Tiene películas en streaming (`Yes`, `No`, `No internet service`).  
* **Contract:** Tipo de contrato (`Month-to-month`, `One year`, `Two year`).  
* **PaperlessBilling:** Usa facturación electrónica (`1`: Sí, `0`: No).  
* **PaymentMethod:** Método de pago del cliente (`Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)`).  
* **MonthlyCharges:** Cargo mensual del cliente (USD).  
* **TotalCharges:** Total facturado acumulado (USD).  
* **Churn (Target):** Cliente abandonó el servicio (`1`: Sí, `0`: No).

**Cita y Referencia:**

El conjunto de datos utilizado en este proyecto proviene del repositorio *Heart Failure Prediction Dataset* disponible en **Kaggle**. La referencia bibliográfica para este trabajo es:

* blastchar. (2019). *Telco Customer Churn*. Recuperado el 27 de abril de 2026 de https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Arquitectura y tecnologías utilizadas

Para garantizar la reproducibilidad y el despliegue continuo, el proyecto hace uso de las siguientes herramientas:

* **Scikit-Learn & Pandas:** Construcción del pipeline de preprocesamiento y entrenamiento del clasificador (Árboles de Decisión).
* **MyST (Markedly Structured Text):** Compilación y renderizado de la documentación estática a partir de libretas Jupyter (`_build`).
* **FastAPI & Pydantic:** Creación del servidor web y validación de esquemas de datos para inferencia.
* **Docker:** Contenerización del entorno de la API y sus dependencias.

# Estructura del repositorio

El proyecto está modularizado para separar claramente la experimentación, los datos, el código fuente y la infraestructura de despliegue:

```text
telco-churn-mlops/
├── .github/
├── _build/
├── app/
│   ├── __init__.py
│   ├── api.py
│   └── model.joblib
├── data/
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_interpretability.ipynb
├── test/
│   ├── __init__.py
│   ├── test_api.py
│   └── test_model.py
├── results/
│   ├── grid_results_cat_20260508_0543.csv
│   ├── grid_results_dt_20260508_0506.csv
│   ├── grid_results_lgbm_20260508_1946.csv
│   ├── grid_results_xgb_20260508_0525.csv
│   └── grid_results_rf_20260508_0428.csv
├── src/
│   ├── eda_toolkit.py
│   ├── model_evaluation_toolkit.py
│   ├── churn_model_selection_toolkit.py
│   └── visual_diagnostics_toolkit.py
├── .gitignore
├── custom.css
├── LICENSE
├── setup.cfg
└── myst.yml
```

# Selección del modelo y métricas

Dada la naturaleza desbalanceada del conjunto de datos, la métrica rectora del proyecto es el F1-Score, definida como la media armónica entre la precisión y la exhaustividad (Recall):

$$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

Se realizó un benchmarking exhaustivo entre múltiples arquitecturas de árboles, cuyos resultados detallados se encuentran en la carpeta results/. El modelo seleccionado para producción fue CatBoost, tras alcanzar un balance óptimo de generalización y estabilidad bajo un esquema de validación cruzada estratificada de 5 pliegues.

# ¿Cómo ejecutar el proyecto?

Siga estos pasos para replicar el entorno de experimentación y despliegue:

## 1. Preparación del Entorno Local

Instale las dependencias optimizadas para evitar conflictos de versiones (NumPy < 2.0):

```bash
pip install -r docker/requirements.txt
```

## 3. Contenerización con Docker

Construya y ejecute la API de inferencia en un entorno aislado:

```bash
docker build -t telco-churn-api:v1.1 .
docker run -d --name telco-churn-container -p 8000:8000 telco-churn-api:v1.1
```