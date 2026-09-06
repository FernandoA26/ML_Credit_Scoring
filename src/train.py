from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.base import clone

from .config import (
    DATA_PATH, MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH, RANDOM_STATE,
    TEST_SIZE, VALIDATION_SIZE, FIGURES_DIR
)
from .data import load_dataset, split_xy
from .modeling import candidate_models
from .metrics import classification_metrics, choose_threshold_f2

def make_splits(X, y):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    relative_val = VALIDATION_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=relative_val,
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def main(data_path=DATA_PATH):
    df = load_dataset(data_path)
    X, y = split_xy(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    models = candidate_models(y_train)
    validation_results = {}

    print(f"Dataset: {len(df):,} registros")
    print(f"Train: {len(X_train):,} | Validación: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Clase positiva total: {y.mean():.2%}\n")

    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        prob_val = model.predict_proba(X_val)[:, 1]
        validation_results[name] = classification_metrics(y_val, prob_val, threshold=0.5)
        print(
            f"  AUC={validation_results[name]['roc_auc']:.4f} "
            f"Recall={validation_results[name]['recall']:.4f} "
            f"F1={validation_results[name]['f1']:.4f}"
        )

    # Selección por AUC de validación; si hay empate, favorece recall.
    selected_name = max(
        validation_results,
        key=lambda n: (
            validation_results[n]["roc_auc"],
            validation_results[n]["recall"],
        )
    )
    selected_model = models[selected_name]

    prob_val = selected_model.predict_proba(X_val)[:, 1]
    threshold, f2_val = choose_threshold_f2(y_val, prob_val)

    # Validación cruzada del modelo ganador SOLO sobre train.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_validate(
        clone(selected_model),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "roc_auc": "roc_auc",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=1,
    )
    cv_summary = {
        metric: {
            "mean": float(cv_scores[f"test_{metric}"].mean()),
            "std": float(cv_scores[f"test_{metric}"].std()),
        }
        for metric in ["roc_auc", "precision", "recall", "f1"]
    }

    # Reentrenar ganador con train + validación.
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    final_model = clone(selected_model)
    final_model.fit(X_train_full, y_train_full)

    prob_test = final_model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, prob_test, threshold=threshold)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": final_model,
        "threshold": threshold,
        "model_name": selected_name,
        "model_version": "1.0.0",
    }
    joblib.dump(artifact, MODEL_PATH)

    output = {
        "dataset_rows": int(len(df)),
        "target_positive_rate": float(y.mean()),
        "split": {
            "train": int(len(X_train)),
            "validation": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "validation_models": validation_results,
        "selected_model": selected_name,
        "selected_threshold_f2": threshold,
        "validation_f2": f2_val,
        "cross_validation_selected": cv_summary,
        "test_metrics": test_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({
        "real": y_test.reset_index(drop=True),
        "probabilidad_riesgo": prob_test,
        "prediccion": (prob_test >= threshold).astype(int),
    })
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    print("\nModelo ganador:", selected_name)
    print("Umbral F2:", round(threshold, 3))
    print("AUC Test:", round(test_metrics["roc_auc"], 4))
    print("Recall Test:", round(test_metrics["recall"], 4))
    print("F1 Test:", round(test_metrics["f1"], 4))
    print("Modelo guardado en:", MODEL_PATH)
    print("Métricas guardadas en:", METRICS_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    args = parser.parse_args()
    main(Path(args.data))
