import pandas as pd
from src.features import CreditFeatureEngineer

def test_feature_engineering_creates_expected_columns():
    X = pd.DataFrame([{
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
        "endeudamiento_interno": 2000,
    }])

    out = CreditFeatureEngineer().fit_transform(X)
    assert "ingreso_consolidado" in out.columns
    assert "ratio_endeudamiento_total" in out.columns
    assert "solvencia_basica" in out.columns
