from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import joblib
import pandas as pd

# Carga modelo y encoder al iniciar la API
model = joblib.load("src/models/churn.pkl")
encoder = joblib.load("src/models/encoder.pkl")

app = FastAPI(title="Churn Prediction API")

class ChurnInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: Literal['Month-to-month', 'One year', 'Two year'] # ← Esto valida

@app.get("/")
def read_root():
    return {"message": "Churn API Jr funcionando con ML real"}

@app.post("/predict")
def predict_churn(data: ChurnInput):
    try:
        input_df = pd.DataFrame([data.model_dump()]) # ← model_dump() en vez de dict()
        input_df['Contract'] = encoder.transform(input_df['Contract'])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        return {"churn_probability": float(proba), "will_churn": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")