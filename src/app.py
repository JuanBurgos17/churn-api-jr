from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI(title="Churn API Jr", version="1.0")

class ClienteData(BaseModel):
    antiguedad: int
    cargos_mensuales: float
    total_cargos: float
    contrato: str  # "Mes a mes", "Un año", "Dos años"

@app.get("/")
def home():
    return {"status": "API funcionando", "docs": "/docs"}

@app.post("/predict")
def predict_churn(data: ClienteData):
    # Modelo dummy por ahora. Luego metes tu churn.pkl
    score = random.random()
    prediccion = 1 if score > 0.5 else 0
    
    return {
        "churn": prediccion,
        "probabilidad": round(score, 2),
        "input": data.dict()
    }