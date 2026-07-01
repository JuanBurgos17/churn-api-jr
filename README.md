Churn Prediction API - Proyecto Jr ML Engineer

Contexto del Problema
Churn = cancelación de clientes. Las empresas de telecomunicaciones pierden 15-25% de clientes al año. Cada cliente perdido cuesta $500-2000 en CAC.

Objetivo: Crear una API que predice la probabilidad de que un cliente cancele su servicio, para que el equipo de retención actúe antes.

Solución: API REST con modelo de Machine Learning que recibe datos del cliente y devuelve churn_probability + risk_level.
Skill	Evidencia en el Proyecto
API con FastAPI	app.py con 3 endpoints + docs automáticas
Validación Pydantic	Literal previene inputs inválidos
Testing	pytest + TestClient para endpoints
CI/CD	GitHub Actions corre tests en cada push
Deploy	Render con deploy automático desde GitHub
ML Ops	Carga de modelo .pkl, manejo de errores
Documentación	Swagger UI en /docs
Monitoreo	/health endpoint para uptime checks



Stack
Python 3.11
FastAPI
MLflow
Docker
Cómo correr
pip install -r requirements.txt
uvicorn src.app:app --reload