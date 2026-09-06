from __future__ import annotations
import argparse
import json
import joblib
import pandas as pd
from .config import MODEL_PATH

def predict_one(payload: dict, model_path=MODEL_PATH):
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]

    X = pd.DataFrame([payload])
    probability = float(pipeline.predict_proba(X)[:, 1][0])
    prediction = int(probability >= threshold)

    if probability < 0.30:
        risk_level = "bajo"
    elif probability < 0.60:
        risk_level = "medio"
    else:
        risk_level = "alto"

    return {
        "probabilidad_riesgo": round(probability, 6),
        "prediccion": prediction,
        "nivel_riesgo": risk_level,
        "umbral_modelo": round(float(threshold), 4),
        "modelo": artifact["model_name"],
        "version": artifact["model_version"],
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Payload JSON con las variables de entrada.")
    args = parser.parse_args()
    print(json.dumps(predict_one(json.loads(args.json)), ensure_ascii=False, indent=2))
