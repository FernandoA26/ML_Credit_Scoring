from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .config import MINIMUM_INCOME_REFERENCE

class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering reproducible para entrenamiento e inferencia."""

    def __init__(self, minimum_income_reference: float = MINIMUM_INCOME_REFERENCE):
        self.minimum_income_reference = minimum_income_reference

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        df = X.copy()

        income = self._consolidate_income(df)
        df["ingreso_consolidado"] = income["valor"]
        df["fuente_ingreso"] = income["fuente"]

        safe_income = df["ingreso_consolidado"].replace(0, self.minimum_income_reference)
        total_debt = df["endeudamiento_interno"].fillna(0) + df["endeudamiento_externo"].fillna(0)

        df["ratio_endeudamiento_total"] = total_debt / safe_income
        df["ratio_ahorro_ingreso"] = df["ahorros_totales"].fillna(0) / safe_income
        df["ratio_aporte_ingreso"] = df["aportes_totales"].fillna(0) / safe_income
        df["ratio_exposicion_credito"] = df["monto_credito_promedio"].fillna(0) / safe_income

        df["intensidad_crediticia"] = (
            df["creditos_activos"].fillna(0) / (df["creditos_historicos"].fillna(0) + 1)
        )

        denom_exp = (df["edad"].fillna(18) - 18 + 1).clip(lower=1)
        df["experiencia_crediticia_relativa"] = (
            df["antiguedad_crediticia"].fillna(0) / denom_exp
        )

        df["estabilidad_financiera"] = (
            df["ahorros_totales"].fillna(0) + df["aportes_totales"].fillna(0)
        ) / (total_debt + 1)

        df["carga_familiar"] = (
            df["numero_hijos"].fillna(0) / (df["numero_familiares"].fillna(0) + 1)
        )

        df["solvencia_basica"] = safe_income / (total_debt + 1)

        # Evitar duplicar tres fuentes de ingreso una vez consolidado.
        df = df.drop(columns=["ingreso_mensual", "capacidad_pago", "utilidad"], errors="ignore")

        return df.replace([np.inf, -np.inf], np.nan)

    def _consolidate_income(self, df: pd.DataFrame) -> pd.DataFrame:
        ref = self.minimum_income_reference
        ingreso = df["ingreso_mensual"].fillna(0)
        utilidad = df["utilidad"].fillna(0)
        capacidad = df["capacidad_pago"].fillna(0)

        conditions = [
            ingreso >= ref,
            utilidad >= ref,
            capacidad >= ref,
            ingreso > 0,
            utilidad > 0,
            capacidad > 0,
        ]
        values = [ingreso, utilidad, capacidad, ingreso, utilidad, capacidad]
        sources = [
            "ingreso_valido",
            "utilidad_valida",
            "capacidad_valida",
            "ingreso_bajo",
            "utilidad_baja",
            "capacidad_baja",
        ]

        valor = np.select(conditions, values, default=ref).astype(float)
        fuente = np.select(conditions, sources, default="referencia_imputada")
        return pd.DataFrame({"valor": valor, "fuente": fuente}, index=df.index)
