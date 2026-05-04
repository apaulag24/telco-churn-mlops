from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Telco Churn Prediction API", version="1.0")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_churn.joblib")
model      = joblib.load(MODEL_PATH)

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

@app.get("/")
def home():
    return {"status": "Online", "model": "Telco Churn Classifier v1.0"}

@app.post("/predict")
def predict(customer: CustomerData):
    data        = pd.DataFrame([customer.dict()])
    prediction  = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]
    return {
        "churn_prediction": int(prediction),
        "probability":      round(float(probability), 4),
        "label":            "Abandona" if prediction == 1 else "Persiste"
    }
