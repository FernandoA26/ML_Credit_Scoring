from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)
from .config import MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH, FIGURES_DIR

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(PREDICTIONS_PATH)

    y_true = preds["real"].values
    y_prob = preds["probabilidad_riesgo"].values
    y_pred = preds["prediccion"].values

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No riesgoso", "Riesgoso"])
    disp.plot(values_format="d")
    plt.title("Matriz de confusión - Test")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("Curva ROC - Test")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title("Curva Precision-Recall - Test")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=160)
    plt.close()

    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    print(json.dumps(metrics["test_metrics"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
