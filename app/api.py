import os
import joblib
import logging
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Configuración de Logging Profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TelcoChurnMLOps")

EXPECTED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
    "MonthlyCharges", "TotalCharges"
]

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicialización
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
    
    logger.info("Cargando artefactos del modelo...")
    try:
        ml_models["churn_model"] = joblib.load(MODEL_PATH)
        logger.info("Modelo cargado exitosamente.")
    except FileNotFoundError:
        logger.error(f"No se encontró el modelo en: {MODEL_PATH}")
        raise RuntimeError(f"Error crítico: Archivo no encontrado.")
    
    yield 
    
    # Limpieza
    ml_models.clear()
    logger.info("Memoria liberada y servicio detenido.")

app = FastAPI(
    title="Telco Churn MLOps API", 
    version="1.1", # Actualizado por mejoras de estabilidad
    lifespan=lifespan
)

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", 
                "Dependents": "No", "tenure": 1, "PhoneService": "No", 
                "MultipleLines": "No phone service", "InternetService": "DSL", 
                "OnlineSecurity": "No", "OnlineBackup": "Yes", 
                "DeviceProtection": "No", "TechSupport": "No", 
                "StreamingTV": "No", "StreamingMovies": "No", 
                "Contract": "Month-to-month", "PaperlessBilling": "Yes", 
                "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, 
                "TotalCharges": 29.85
            }
        }
    }

@app.get("/")
def home():
    return {"status": "Online", "model": "Telco Churn Classifier v1.1"}

# 2. Endpoint de Salud (Health Check)
@app.get("/health")
def health_check():
    is_ready = "churn_model" in ml_models
    return {
        "status": "ready" if is_ready else "loading",
        "components": {
            "model_loaded": is_ready
        }
    }

@app.post("/predict")
def predict(customer: CustomerData):
    try:
        # 3. Validación de Tipos y Construcción del DataFrame
        # Convertimos a DataFrame y forzamos tipos explícitos para el pipeline
        raw_data = customer.model_dump()
        data = pd.DataFrame([raw_data])[EXPECTED_COLUMNS]
        
        # Aseguramos consistencia de tipos (Crucial para Scikit-Learn/Pandas)
        data = data.astype({
            "SeniorCitizen": int,
            "tenure": int,
            "MonthlyCharges": float,
            "TotalCharges": float
        })

        model = ml_models["churn_model"]
        
        # Inferencia
        logger.info("Ejecutando inferencia para nueva instancia...")
        probability = float(model.predict_proba(data)[0][1])
        umbral_decision = 0.5 
        prediction = int(probability > umbral_decision)
        
        logger.info(f"Predicción completada: {prediction} (Prob: {probability:.4f})")

        return {
            "churn_prediction": prediction,
            "probability": round(probability, 4),
            "label": "Abandonó" if prediction == 1 else "Permaneció"
        }

    except Exception as e:
        logger.error(f"Fallo en la predicción: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Error interno al procesar la solicitud de inferencia."
        )