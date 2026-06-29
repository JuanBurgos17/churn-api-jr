import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# 1. Descarga el dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# 2. Limpieza básica
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Features en la API
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']
X = df[features].copy()
y = df['Churn']

# 4. transformar los datos de la columna contrato de categoricos a numericos
le = LabelEncoder() # asigna por orden valores desde el 0 .... para modelos mas avanzados se sugiere modificar por One-Hot Encoding
X['Contract'] = le.fit_transform(X['Contract'])

# 5. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. MLflow Tracking
mlflow.set_experiment("Churn-Prediction")
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # Log en MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"Accuracy: {acc:.3f} | F1: {f1:.3f}")

# 7. Guardar modelo y encoder
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/churn.pkl")
joblib.dump(le, "models/encoder.pkl")
print("Modelo guardado en models/churn.pkl")

