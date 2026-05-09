from fastapi.testclient import TestClient
from app.api import app

def test_health_endpoint():
    """Verifica que el servicio y el modelo estén listos."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["components"]["model_loaded"] is True


def test_predict_endpoint():
    """Prueba la integridad de la inferencia con el payload del pipeline."""
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

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        result = response.json()
        assert "churn_prediction" in result
        assert "probability" in result
        assert result["label"] in ["Abandonó", "Permaneció"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])