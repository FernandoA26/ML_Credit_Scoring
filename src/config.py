from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "credit_scoring_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "credit_scoring_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
PREDICTIONS_PATH = PROJECT_ROOT / "reports" / "test_predictions.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
MINIMUM_INCOME_REFERENCE = 1130.0

TARGET_NAME = "target_riesgo"

# Se usan solamente para construir el target durante el entrenamiento.
# NUNCA deben entrar al modelo como predictores.
TARGET_SOURCE_COLUMNS = [
    "atrasos_12m",
    "ratio_mora_actual",
    "calificacion_interna_max",
]

MODEL_INPUT_COLUMNS = [
    "edad",
    "estado_civil_codigo",
    "numero_hijos",
    "numero_familiares",
    "antiguedad_cliente",
    "antiguedad_crediticia",
    "creditos_historicos",
    "creditos_activos",
    "calificacion_externa_max",
    "aportes_totales",
    "ahorros_totales",
    "monto_credito_promedio",
    "tasa_promedio",
    "plazo_promedio",
    "endeudamiento_externo",
    "numero_entidades",
    "calificacion_externa",
    "calificacion_sistema_12m",
    "ingreso_mensual",
    "capacidad_pago",
    "utilidad",
    "endeudamiento_interno",
]
