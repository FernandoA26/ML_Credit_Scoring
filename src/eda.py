from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DATA_PATH, PROJECT_ROOT, TARGET_SOURCE_COLUMNS
from .data import build_target, load_dataset


DEFAULT_EDA_DIR = PROJECT_ROOT / "reports" / "eda"

# Variables numéricas principales del dataset de Credit Scoring.
KEY_NUMERIC_COLUMNS = [
    "edad",
    "numero_hijos",
    "numero_familiares",
    "antiguedad_cliente",
    "antiguedad_crediticia",
    "creditos_historicos",
    "creditos_activos",
    "atrasos_12m",
    "ratio_mora_actual",
    "calificacion_interna_max",
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

# Variables discretas/categóricas relevantes.
KEY_CATEGORICAL_COLUMNS = [
    "estado_civil_codigo",
    "calificacion_interna_max",
    "calificacion_externa_max",
    "calificacion_externa",
    "calificacion_sistema_12m",
]


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Crea la estructura de carpetas usada por el EDA."""
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    distributions_dir = figures_dir / "distribuciones"
    boxplots_dir = figures_dir / "boxplots_target"
    categories_dir = figures_dir / "categorias_target"

    for directory in [
        output_dir,
        figures_dir,
        tables_dir,
        distributions_dir,
        boxplots_dir,
        categories_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        "root": output_dir,
        "figures": figures_dir,
        "tables": tables_dir,
        "distributions": distributions_dir,
        "boxplots": boxplots_dir,
        "categories": categories_dir,
    }


def dataset_overview(df: pd.DataFrame, y: pd.Series) -> dict:
    """Resumen general del dataset y de la variable objetivo."""
    counts = y.value_counts().sort_index()
    rates = y.value_counts(normalize=True).sort_index()

    return {
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "duplicados": int(df.duplicated().sum()),
        "valores_nulos_totales": int(df.isna().sum().sum()),
        "target": {
            "no_riesgoso_0": int(counts.get(0, 0)),
            "riesgoso_1": int(counts.get(1, 0)),
            "proporcion_no_riesgoso": float(rates.get(0, 0.0)),
            "proporcion_riesgoso": float(rates.get(1, 0.0)),
        },
    }


def variable_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Calidad de datos por variable."""
    rows = []

    for column in df.columns:
        series = df[column]

        rows.append(
            {
                "variable": column,
                "tipo": str(series.dtype),
                "nulos": int(series.isna().sum()),
                "porcentaje_nulos": float(series.isna().mean() * 100),
                "valores_unicos": int(series.nunique(dropna=True)),
                "porcentaje_unicos": float(
                    series.nunique(dropna=True) / len(series) * 100
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["porcentaje_nulos", "valores_unicos"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos descriptivos ampliados de variables numéricas."""
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    stats = numeric.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T

    stats["nulos"] = numeric.isna().sum()
    stats["asimetria"] = numeric.skew(numeric_only=True)
    stats["curtosis"] = numeric.kurt(numeric_only=True)

    return stats.reset_index().rename(columns={"index": "variable"})


def categorical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de variables categóricas/discretas seleccionadas."""
    rows = []

    for column in KEY_CATEGORICAL_COLUMNS:
        if column not in df.columns:
            continue

        series = df[column]
        mode = series.mode(dropna=True)

        rows.append(
            {
                "variable": column,
                "valores_unicos": int(series.nunique(dropna=True)),
                "moda": None if mode.empty else str(mode.iloc[0]),
                "frecuencia_moda": (
                    0 if mode.empty else int((series == mode.iloc[0]).sum())
                ),
                "nulos": int(series.isna().sum()),
                "porcentaje_nulos": float(series.isna().mean() * 100),
            }
        )

    return pd.DataFrame(rows)


def outlier_iqr_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta valores atípicos mediante IQR:
      LI = Q1 - 1.5 * IQR
      LS = Q3 + 1.5 * IQR
    """
    rows = []

    for column in KEY_NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((series < lower) | (series > upper)).sum()

        rows.append(
            {
                "variable": column,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "limite_inferior": float(lower),
                "limite_superior": float(upper),
                "cantidad_outliers": int(outliers),
                "porcentaje_outliers": float(outliers / len(series) * 100),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("porcentaje_outliers", ascending=False)
        .reset_index(drop=True)
    )


def correlation_matrix(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Matriz de correlación de Pearson de todas las variables numéricas.
    El target se añade solo para análisis exploratorio.
    """
    numeric = df.select_dtypes(include=np.number).copy()
    numeric["target_riesgo"] = y.values
    return numeric.corr(numeric_only=True)


def target_correlations(df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Correlación de cada variable numérica con el target.
    Equivale conceptualmente al corrwith(y) utilizado en el notebook.
    """
    numeric = df.select_dtypes(include=np.number).copy()

    corr = numeric.corrwith(y).dropna()

    result = (
        corr.rename("correlacion_target")
        .reset_index()
        .rename(columns={"index": "variable"})
    )

    result["correlacion_absoluta"] = result["correlacion_target"].abs()
    result["es_fuente_del_target"] = result["variable"].isin(TARGET_SOURCE_COLUMNS)

    return (
        result.sort_values("correlacion_absoluta", ascending=False)
        .reset_index(drop=True)
    )


def high_correlation_pairs(
    df: pd.DataFrame,
    threshold: float = 0.80,
) -> pd.DataFrame:
    """
    Identifica pares de variables con correlación absoluta >= threshold.

    Esta función replica el análisis de colinealidad del notebook.
    Solo evalúa variables numéricas y usa la parte superior de la matriz
    para evitar duplicar pares.
    """
    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:
        return pd.DataFrame(
            columns=[
                "variable_1",
                "variable_2",
                "correlacion",
                "correlacion_absoluta",
            ]
        )

    corr = numeric.corr(numeric_only=True)

    upper_mask = np.triu(
        np.ones(corr.shape, dtype=bool),
        k=1,
    )

    rows = []

    for i, row_name in enumerate(corr.index):
        for j, col_name in enumerate(corr.columns):
            if not upper_mask[i, j]:
                continue

            value = corr.iloc[i, j]

            if pd.notna(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "variable_1": row_name,
                        "variable_2": col_name,
                        "correlacion": float(value),
                        "correlacion_absoluta": float(abs(value)),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "variable_1",
                "variable_2",
                "correlacion",
                "correlacion_absoluta",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values("correlacion_absoluta", ascending=False)
        .reset_index(drop=True)
    )


def save_target_distribution(y: pd.Series, figures_dir: Path) -> None:
    """Gráfico de distribución del target."""
    counts = y.value_counts().sort_index()
    labels = ["No riesgoso", "Riesgoso"]
    values = [counts.get(0, 0), counts.get(1, 0)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values)

    ax.set_title("Distribución de la variable objetivo")
    ax.set_ylabel("Número de registros")

    total = len(y)

    for bar, value in zip(bars, values):
        pct = value / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(figures_dir / "01_distribucion_target.png", dpi=160)
    plt.close(fig)


def save_missing_values_plot(df: pd.DataFrame, figures_dir: Path) -> None:
    """Gráfico del porcentaje de nulos por variable."""
    missing = (df.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    if missing.empty:
        ax.text(
            0.5,
            0.5,
            "No se identificaron valores nulos",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        ax.barh(missing.index[::-1], missing.values[::-1])
        ax.set_xlabel("Porcentaje de valores nulos")
        ax.set_title("Valores nulos por variable")

    fig.tight_layout()
    fig.savefig(figures_dir / "02_valores_nulos.png", dpi=160)
    plt.close(fig)


def save_correlation_heatmap(
    corr: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Guarda la matriz general de correlación.
    Corresponde a la matriz de correlación mostrada en el EDA del notebook.
    """
    fig_size = max(10, min(18, len(corr.columns) * 0.65))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    image = ax.imshow(
        corr.values,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Matriz de correlación de variables numéricas")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    fig.savefig(
        figures_dir / "03_matriz_correlacion.png",
        dpi=170,
    )
    plt.close(fig)


def save_target_correlations_plot(
    corr_target: pd.DataFrame,
    figures_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Gráfico de las variables con mayor correlación absoluta con el target.
    """
    plot_df = corr_target.head(top_n).copy()
    plot_df = plot_df.sort_values("correlacion_target")

    fig, ax = plt.subplots(figsize=(10, max(6, len(plot_df) * 0.35)))

    ax.barh(
        plot_df["variable"],
        plot_df["correlacion_target"],
    )

    ax.axvline(0, linewidth=1)
    ax.set_title(
        f"Top {min(top_n, len(plot_df))} correlaciones con el target"
    )
    ax.set_xlabel("Correlación de Pearson con target_riesgo")

    fig.tight_layout()

    fig.savefig(
        figures_dir / "04_correlacion_variables_vs_target.png",
        dpi=170,
    )
    plt.close(fig)


def save_high_correlation_heatmap(
    df: pd.DataFrame,
    figures_dir: Path,
    threshold: float = 0.80,
) -> None:
    """
    Matriz visual de correlaciones fuertes para análisis de colinealidad.
    Replica la idea del gráfico '> 0.8 (colinealidad)' del notebook.
    """
    numeric = df.select_dtypes(include=np.number)
    corr = numeric.corr(numeric_only=True)

    # Conservamos solo correlaciones fuertes fuera de la diagonal.
    strong = corr.copy()
    mask = (strong.abs() < threshold) | np.eye(len(strong), dtype=bool)
    strong = strong.mask(mask)

    fig_size = max(10, min(18, len(corr.columns) * 0.65))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    matrix = np.ma.masked_invalid(strong.values)

    image = ax.imshow(
        matrix,
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)

    ax.set_title(
        f"Correlaciones fuertes |r| >= {threshold:.2f} (colinealidad)"
    )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    fig.savefig(
        figures_dir / "05_correlacion_alta_colinealidad.png",
        dpi=170,
    )
    plt.close(fig)


def save_numeric_distributions(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Histogramas de variables numéricas."""
    for column in KEY_NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if series.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(series, bins=30)
        ax.set_title(f"Distribución de {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frecuencia")

        fig.tight_layout()
        fig.savefig(output_dir / f"{column}.png", dpi=140)
        plt.close(fig)


def save_boxplots_by_target(
    df: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> None:
    """Boxplots de variables numéricas comparadas por target."""
    for column in KEY_NUMERIC_COLUMNS:
        if column not in df.columns:
            continue

        values_0 = pd.to_numeric(
            df.loc[y == 0, column],
            errors="coerce",
        ).dropna()

        values_1 = pd.to_numeric(
            df.loc[y == 1, column],
            errors="coerce",
        ).dropna()

        if values_0.empty or values_1.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(
            [values_0, values_1],
            tick_labels=["No riesgoso", "Riesgoso"],
            showfliers=False,
        )

        ax.set_title(f"{column} según condición de riesgo")
        ax.set_ylabel(column)

        fig.tight_layout()
        fig.savefig(output_dir / f"{column}.png", dpi=140)
        plt.close(fig)


def save_target_by_category(
    df: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> None:
    """Tasa de riesgo observada por variable categórica/discreta."""
    temp = df.copy()
    temp["target_riesgo"] = y.values

    for column in KEY_CATEGORICAL_COLUMNS:
        if column not in temp.columns:
            continue

        grouped = (
            temp.groupby(column, dropna=False)["target_riesgo"]
            .agg(["count", "mean"])
            .sort_values("count", ascending=False)
            .head(15)
        )

        if grouped.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.bar(
            grouped.index.astype(str),
            grouped["mean"] * 100,
        )

        ax.set_title(f"Tasa de riesgo por {column}")
        ax.set_ylabel("Clientes riesgosos (%)")
        ax.set_xlabel(column)
        ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        fig.savefig(output_dir / f"{column}.png", dpi=140)
        plt.close(fig)


def target_source_diagnostic(
    corr_target: pd.DataFrame,
) -> pd.DataFrame:
    """
    Separa las variables utilizadas para construir el target.

    Su correlación con el target puede ser alta por definición.
    Este reporte sirve para documentar por qué se excluyen del modelo
    y evitar data leakage.
    """
    result = corr_target[
        corr_target["variable"].isin(TARGET_SOURCE_COLUMNS)
    ].copy()

    return result.reset_index(drop=True)


def run_eda(
    data_path: Path = DATA_PATH,
    output_dir: Path = DEFAULT_EDA_DIR,
    correlation_threshold: float = 0.80,
    top_target_correlations: int = 20,
) -> None:
    """
    Ejecuta el EDA completo y guarda resultados tabulares y gráficos.
    """
    dirs = ensure_output_dirs(output_dir)

    print(f"Cargando dataset: {data_path}")

    df = load_dataset(data_path)
    y = build_target(df)

    # 1. Resumen y calidad
    overview = dataset_overview(df, y)
    quality = variable_quality_table(df)
    numeric_stats = numeric_statistics(df)
    categorical_stats = categorical_statistics(df)
    outliers = outlier_iqr_table(df)

    # 2. Correlaciones
    corr_matrix = correlation_matrix(df, y)
    corr_target = target_correlations(df, y)
    high_corr = high_correlation_pairs(
        df,
        threshold=correlation_threshold,
    )
    target_sources = target_source_diagnostic(corr_target)

    # 3. Guardado de tablas
    with open(
        dirs["root"] / "resumen_eda.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overview,
            file,
            ensure_ascii=False,
            indent=2,
        )

    quality.to_csv(
        dirs["tables"] / "calidad_variables.csv",
        index=False,
        encoding="utf-8",
    )

    numeric_stats.to_csv(
        dirs["tables"] / "estadisticos_numericos.csv",
        index=False,
        encoding="utf-8",
    )

    categorical_stats.to_csv(
        dirs["tables"] / "estadisticos_categoricos.csv",
        index=False,
        encoding="utf-8",
    )

    outliers.to_csv(
        dirs["tables"] / "outliers_iqr.csv",
        index=False,
        encoding="utf-8",
    )

    corr_matrix.to_csv(
        dirs["tables"] / "matriz_correlacion.csv",
        encoding="utf-8",
    )

    corr_target.to_csv(
        dirs["tables"] / "correlacion_con_target.csv",
        index=False,
        encoding="utf-8",
    )

    high_corr.to_csv(
        dirs["tables"] / "variables_alta_correlacion.csv",
        index=False,
        encoding="utf-8",
    )

    target_sources.to_csv(
        dirs["tables"] / "variables_fuente_target.csv",
        index=False,
        encoding="utf-8",
    )

    # 4. Gráficos principales
    save_target_distribution(
        y,
        dirs["figures"],
    )

    save_missing_values_plot(
        df,
        dirs["figures"],
    )

    save_correlation_heatmap(
        corr_matrix,
        dirs["figures"],
    )

    save_target_correlations_plot(
        corr_target,
        dirs["figures"],
        top_n=top_target_correlations,
    )

    save_high_correlation_heatmap(
        df,
        dirs["figures"],
        threshold=correlation_threshold,
    )

    # 5. Gráficos complementarios
    save_numeric_distributions(
        df,
        dirs["distributions"],
    )

    save_boxplots_by_target(
        df,
        y,
        dirs["boxplots"],
    )

    save_target_by_category(
        df,
        y,
        dirs["categories"],
    )

    # 6. Resumen en consola
    print("\nEDA COMPLETADO")
    print("-" * 70)
    print(f"Registros: {overview['filas']:,}")
    print(f"Columnas: {overview['columnas']}")
    print(f"Duplicados: {overview['duplicados']:,}")
    print(
        "Tasa de clientes riesgosos: "
        f"{overview['target']['proporcion_riesgoso']:.2%}"
    )
    print(
        "Pares con alta correlación "
        f"(|r| >= {correlation_threshold:.2f}): "
        f"{len(high_corr)}"
    )

    if not target_sources.empty:
        print("\nVariables usadas en la construcción del target:")
        for _, row in target_sources.iterrows():
            print(
                f"  - {row['variable']}: "
                f"corr={row['correlacion_target']:.4f}"
            )

    print(f"\nResultados guardados en: {dirs['root']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EDA completo - Proyecto Credit Scoring"
    )

    parser.add_argument(
        "--data",
        default=str(DATA_PATH),
        help="Ruta del CSV de entrada.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_EDA_DIR),
        help="Directorio donde se guardará el EDA.",
    )

    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.80,
        help="Umbral absoluto para detectar colinealidad.",
    )

    parser.add_argument(
        "--top-target-corr",
        type=int,
        default=20,
        help="Número de variables a mostrar en el gráfico de correlación con target.",
    )

    args = parser.parse_args()

    run_eda(
        data_path=Path(args.data),
        output_dir=Path(args.output_dir),
        correlation_threshold=args.corr_threshold,
        top_target_correlations=args.top_target_corr,
    )
