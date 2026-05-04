# Prediccion de Abandono de Clientes (Telco Churn)

Este proyecto desarrolla un modelo de clasificacion con CatBoost para predecir el abandono de clientes.

## Detalles Tecnicos
* Modelo: CatBoost Classifier
* Optimizacion: Grid Search (5-fold CV)
* Metrica: F1 Score 0.5850

## Estructura
* data/: Dataset original
* notebooks/: EDA y Entrenamiento
* app/: API y Modelo guardado (joblib)
