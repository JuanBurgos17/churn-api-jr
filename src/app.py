from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import joblib
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
import time
from fastapi import Request

# Configurar Loguru: borra el default y agrega formato bonito
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)

# Rutas relativas para que funcione en Render y local
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn.pkl"
ENCODER_PATH = BASE_DIR / "models" / "encoder.pkl"

# Carga modelo y encoder al iniciar la API
try:
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    raise RuntimeError(f"Error cargando modelo o encoder: {e}")

app = FastAPI(
    title="Churn Prediction API",
    description="API para predecir si un cliente cancelará el servicio",
    version="1.0.0"
)

class ChurnInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: Literal['Month-to-month', 'One year', 'Two year']

    class Config:
        json_schema_extra = {
            "example": {
                "tenure": 12,
                "MonthlyCharges": 70.5,
                "TotalCharges": 846.0,
                "Contract": "Month-to-month"
            }
        }

@app.get("/")
def read_root():
    return {"message": "Churn API Jr funcionando con ML real"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_churn(data: ChurnInput):
    try:
        logger.info(f"Predicción iniciada | tenure={data.tenure} | contract={data.Contract}")

        input_df = pd.DataFrame([data.model_dump()])
        input_df['Contract'] = encoder.transform(input_df['Contract'])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        risk = "high" if proba > 0.7 else "medium" if proba > 0.4 else "low"
        logger.success(f"Predicción OK | prob={proba:.4f} | risk={risk}")

        return {
            "churn_probability": round(float(proba), 4),
            "will_churn": bool(prediction),
            "risk_level": risk
        }
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")
    
@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path in ["/docs", "/openapi.json", "/health"]:
        return await call_next(request)
    
    start_time = time.time()
    response = await call_next(request)
    process_time = round((time.time() - start_time) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} | Status: {response.status_code} | Time: {process_time}ms")
    return response