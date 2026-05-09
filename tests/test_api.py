import requests

BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Verifica que el servicio y el modelo estén listos."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    print("Health Check: Pasó")

def test_predict_endpoint():
    """Prueba la inferencia con el payload completo requerido por el pipeline."""
    payload = {
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
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    
    result = response.json()
    assert "churn_prediction" in result
    assert "probability" in result
    print(f"Predicción API: Pasó (Prob: {result['probability']})")

if __name__ == "__main__":
    print("--- Iniciando Pruebas de API ---")
    try:
        test_health_endpoint()
        test_predict_endpoint()
    except Exception as e:
        print(f"Error en las pruebas: {e}")