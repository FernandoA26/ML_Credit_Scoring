from __future__ import annotations
import pandas as pd
from .config import TARGET_SOURCE_COLUMNS, MODEL_INPUT_COLUMNS, TARGET_NAME

def load_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df)
    return df

def validate_columns(df: pd.DataFrame) -> None:
    required = set(TARGET_SOURCE_COLUMNS + MODEL_INPUT_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Reproduce la definición usada en el proyecto original:
      1 = cumple al menos uno:
          - atrasos_12m >= 2
          - calificacion_interna_max >= 3
          - ratio_mora_actual > 0.05
      0 = caso contrario.

    IMPORTANTE:
    estas tres variables quedan excluidas de X para evitar data leakage.
    """
    y = (
        (df["atrasos_12m"] >= 2)
        | (df["calificacion_interna_max"] >= 3)
        | (df["ratio_mora_actual"] > 0.05)
    ).astype("int8")
    y.name = TARGET_NAME
    return y

def split_xy(df: pd.DataFrame):
    y = build_target(df)
    X = df[MODEL_INPUT_COLUMNS].copy()
    return X, y
