# Credit Scoring con Machine Learning

Proyecto académico para estimar riesgo crediticio mediante modelos supervisados de clasificación.

## Arquitectura

1. `data/credit_scoring_data.csv`: dataset de trabajo con cabeceras neutrales y sin identificadores directos.
2. `src/data.py`: carga, validación y construcción del target.
3. `src/features.py`: ingeniería de características reproducible.
4. `src/modeling.py`: pipelines y modelos candidatos.
5. `src/train.py`: partición Train/Validation/Test, comparación, selección, CV y persistencia.
6. `src/evaluate.py`: matriz de confusión, ROC y Precision-Recall.
7. `src/predict.py`: inferencia individual.
8. `api/main.py`: API REST con FastAPI.
9. `load_tests/locustfile.py`: prueba de carga.

## Consideración metodológica crítica

El target reproduce la lógica del notebook original:

- `atrasos_12m >= 2`, o
- `calificacion_interna_max >= 3`, o
- `ratio_mora_actual > 0.05`.

Estas variables **solo se usan para construir el target y se excluyen del modelo** para evitar fuga directa de información.

Para una segunda versión orientada a Probability of Default real, se recomienda construir un target temporal:
usar variables disponibles en fecha T y definir incumplimiento durante T+3/T+6/T+12.

## Seguridad de datos

Cambiar nombres de columnas no anonimiza por completo un dataset. Por eso, esta versión también elimina los identificadores internos directos. El archivo real debe mantenerse fuera de GitHub; `.gitignore` ya lo excluye.

## Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Entrenamiento

Desde la raíz del proyecto:

```bash
python -m src.train
```

Se generan:

- `models/credit_scoring_pipeline.joblib`
- `reports/metrics.json`
- `reports/test_predictions.csv`

La estrategia es:

- 70% entrenamiento
- 15% validación
- 15% test

La selección del modelo se hace en validación por ROC-AUC, usando Recall como desempate.
El umbral se elige en validación maximizando F2 para dar mayor peso a la detección de casos riesgosos.

## Evaluación gráfica

```bash
python -m src.evaluate
```

Genera gráficos en `reports/figures/`.

## API

```bash
uvicorn api.main:app --reload
```

Abrir:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Docker

```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
```

## Pruebas

```bash
pytest
```

## Prueba de carga

Con la API ejecutándose:

```bash
locust -f load_tests/locustfile.py --host http://127.0.0.1:8000
```

## Qué se debe subir a GitHub

Sí:
- `src/`
- `api/`
- `tests/`
- `load_tests/`
- `README.md`
- `requirements.txt`
- `Dockerfile`
- notebook limpio sin información sensible

No:
- dataset real
- modelo entrenado si contiene artefactos derivados de datos privados
- archivos con identificadores o rutas internas
