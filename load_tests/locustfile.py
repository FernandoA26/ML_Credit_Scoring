from locust import HttpUser, task, between

SAMPLE = {
    "edad": 35,
    "estado_civil_codigo": 1,
    "numero_hijos": 2,
    "numero_familiares": 4,
    "antiguedad_cliente": 5,
    "antiguedad_crediticia": 7,
    "creditos_historicos": 4,
    "creditos_activos": 2,
    "calificacion_externa_max": 1,
    "aportes_totales": 500,
    "ahorros_totales": 1500,
    "monto_credito_promedio": 6000,
    "tasa_promedio": 2.0,
    "plazo_promedio": 18,
    "endeudamiento_externo": 1000,
    "numero_entidades": 2,
    "calificacion_externa": 1,
    "calificacion_sistema_12m": 1,
    "ingreso_mensual": 2500,
    "capacidad_pago": 1000,
    "utilidad": 1800,
    "endeudamiento_interno": 2000
}

class CreditScoringUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def predict(self):
        self.client.post("/predict", json=SAMPLE)
