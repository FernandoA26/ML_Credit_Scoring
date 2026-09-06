from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from .features import CreditFeatureEngineer
from .config import RANDOM_STATE

CATEGORICAL_FEATURES = [
    "estado_civil_codigo",
    "calificacion_externa_max",
    "calificacion_externa",
    "calificacion_sistema_12m",
    "fuente_ingreso",
]

NUMERIC_FEATURES = [
    "edad",
    "numero_hijos",
    "numero_familiares",
    "antiguedad_cliente",
    "antiguedad_crediticia",
    "creditos_historicos",
    "creditos_activos",
    "aportes_totales",
    "ahorros_totales",
    "monto_credito_promedio",
    "tasa_promedio",
    "plazo_promedio",
    "endeudamiento_externo",
    "numero_entidades",
    "endeudamiento_interno",
    "ingreso_consolidado",
    "ratio_endeudamiento_total",
    "ratio_ahorro_ingreso",
    "ratio_aporte_ingreso",
    "ratio_exposicion_credito",
    "intensidad_crediticia",
    "experiencia_crediticia_relativa",
    "estabilidad_financiera",
    "carga_familiar",
    "solvencia_basica",
]

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ], remainder="drop")

def build_pipeline(classifier):
    return Pipeline([
        ("features", CreditFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("classifier", classifier),
    ])

def candidate_models(y_train):
    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = max(int(np.sum(y_train == 0)), 1)
    scale_pos_weight = negatives / positives

    return {
        "regresion_logistica": build_pipeline(
            LogisticRegression(
                solver="liblinear",
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features=0.5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "xgboost": build_pipeline(
            XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="auc",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }
