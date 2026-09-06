from __future__ import annotations
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_one

app = FastAPI(
    title="Credit Scoring ML API",
    version="1.0.0",
    description="API académica para inferencia de riesgo crediticio."
)

class CreditApplication(BaseModel):
    edad: float
    estado_civil_codigo: int
    numero_hijos: float = 0
    numero_familiares: float = 0
    antiguedad_cliente: float = 0
    antiguedad_crediticia: float = 0
    creditos_historicos: float = 0
    creditos_activos: float = 0
    calificacion_externa_max: int = 0
    aportes_totales: float = 0
    ahorros_totales: float = 0
    monto_credito_promedio: float = 0
    tasa_promedio: float = 0
    plazo_promedio: float = 0
    endeudamiento_externo: float = 0
    numero_entidades: float = 0
    calificacion_externa: int = 0
    calificacion_sistema_12m: int = 0
    ingreso_mensual: float = 0
    capacidad_pago: float = 0
    utilidad: float = 0
    endeudamiento_interno: float = 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(application: CreditApplication):
    try:
        return predict_one(application.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
