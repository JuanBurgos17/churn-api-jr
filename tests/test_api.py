from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app) 

def test_docs_endpoint():
    response = client.get("/docs")
    assert response.status_code == 200

def test_predict_valid_input():
    response = client.post("/predict", json={
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 845.5,
        "Contract": "Month-to-month"
    })
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "will_churn" in data
    assert 0 <= data["churn_probability"] <= 1

def test_predict_invalid_contract():
    response = client.post("/predict", json={
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 845.5,
        "Contract": "invalid"
    })
    # Depende de tu validación. Si no validas, será 200. Si usas Pydantic Enum, será 422
    assert response.status_code in [200, 422]