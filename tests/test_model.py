import joblib
import pandas as pd
import os

def test_model_serialization():
    """Carga el modelo directamente y verifica una inferencia básica."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "../app/model_churn.joblib")
    
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el modelo en {model_path}")
        return

    model = joblib.load(model_path)
    columns = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", 
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", 
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", 
        "MonthlyCharges", "TotalCharges"
    ]
    
    sample_data = pd.DataFrame([["Male", 0, "No", "No", 12, "Yes", "No", "Fiber optic", 
                                 "No", "No", "No", "No", "No", "No", "Month-to-month", 
                                 "Yes", "Electronic check", 70.0, 840.0]], columns=columns)
    
    try:
        proba = model.predict_proba(sample_data)[0][1]
        print(f"Validación del Artefacto: Pasó (Probabilidad calculada: {proba:.4f})")
    except Exception as e:
        print(f"Error en la lógica del modelo: {e}")

if __name__ == "__main__":
    print("--- Iniciando Validación del Modelo ---")
    test_model_serialization()