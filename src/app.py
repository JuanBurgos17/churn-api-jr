from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Carga modelo y encoder al iniciar la API
model = joblib.load("models/churn.pkl")
encoder = joblib.load("models/encoder.pkl")

app = FastAPI(title="Churn Prediction API")

class ChurnInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str # "Month-to-month", "One year", "Two year"

@app.get("/")
def read_root():
    return {"message": "Churn API Jr funcionando con ML real"}

@app.post("/predict")
def predict_churn(data: ChurnInput):
    # 1. Convertir input a DataFrame
    input_df = pd.DataFrame([data.dict()])

    # 2. Encodear Contract igual que en training
    input_df['Contract'] = encoder.transform(input_df['Contract'])

    # 3. Predecir probabilidad de churn
    prob = model.predict_proba(input_df)[0][1] # Probabilidad de clase 1 = Churn

    return {
        "churn_probability": round(float(prob), 3),
        "will_churn": bool(prob > 0.5)
    }
