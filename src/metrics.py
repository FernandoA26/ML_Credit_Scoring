from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve
)

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = float(np.max(tpr - fpr))
    auc = float(roc_auc_score(y_true, y_prob))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "ks": ks,
        "gini": float(2 * auc - 1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def choose_threshold_f2(y_true, y_prob):
    """
    Selecciona el umbral en VALIDACIÓN maximizando F2,
    dando mayor peso al recall de clientes riesgosos.
    """
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.arange(0.10, 0.91, 0.01):
        pred = (np.asarray(y_prob) >= threshold).astype(int)
        score = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)
