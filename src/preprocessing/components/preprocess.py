# src/preprocessing/components/preprocess.py

import argparse
import logging
import time
import os

import numpy as np
import pandas as pd

from preprocessing.config.configuration import ConfigurationManager
from preprocessing.repository.repository import CsvPreprocessingRepository

# Prometheus (minimal push)
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Pushgateway URL (Compose service name works on the Docker network)
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
PUSH_GROUPING = {"instance": os.getenv("INSTANCE", "local"), "service": "preprocessing"}


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SKU"] = df["SKU"].astype(str)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    numeric_cols = [
        "Prix",
        "PrixInitial",
        "AgeProduitEnJours",
        "QuantiteVendue",
        "UtiliteProduit",
        "ElasticitePrix",
        "Remise",
        "Qualite",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Timestamp" in df.columns:
        ts = df["Timestamp"]
        df["Mois_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
        df["Mois_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
        df["Heure_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["Heure_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [
        "DateLancement",
        "PrixPlancher",
        "PlancherPourcentage",
        "ErreurAleatoire",
        "Categorie",
        "Promotion",
    ]
    existing = [c for c in to_drop if c in df.columns]
    return df.drop(columns=existing, errors="ignore")


def _push_metrics(status: int, duration_s: float, rows: int) -> None:
    """Push a few gauges to Pushgateway; never raise."""
    try:
        reg = CollectorRegistry()
        g_status = Gauge("job_status", "1=success,0=failure", registry=reg)
        g_dur = Gauge("job_duration_seconds", "Job wall time in seconds", registry=reg)
        g_rows = Gauge(
            "rows_processed_total", "Rows processed by the job", registry=reg
        )

        g_status.set(float(status))
        g_dur.set(float(duration_s))
        g_rows.set(float(rows))

        push_to_gateway(
            PUSHGATEWAY_URL,
            job="preprocessing",
            grouping_key=PUSH_GROUPING,
            registry=reg,
        )
        logger.info(
            f"Pushed metrics to Pushgateway ({PUSHGATEWAY_URL}) "
            f"status={status} duration_s={duration_s:.3f} rows={rows}"
        )
    except Exception as e:
        logger.warning(f"Failed to push metrics to Pushgateway: {e}")


def run_preprocessing(config_path: str, params_path: str):
    t0 = time.perf_counter()
    rows_out = 0
    status = 0  # pessimistic default

    try:
        # 1) Chargement des configs et params
        cm = ConfigurationManager(config_path, params_path)
        cfg = cm.get_preprocessing_config()
        params = cm.get_params()

        # 2) Lecture des données brutes
        df = pd.read_csv(cfg.raw_data_path, encoding="utf-8")
        logger.info(f"Charged raw data: {cfg.raw_data_path} (shape={df.shape})")

        # 3) Transformations séquentielles
        df = convert_types(df)
        df = add_time_features(df)
        df = drop_unused_columns(df)

        # 4) Sélection des colonnes finales
        keep = params.columns_to_keep
        df_clean = df[keep].copy()
        rows_out = len(df_clean)
        logger.info(f"Columns kept: {keep} (rows={rows_out})")

        # 5) Persistance via repository
        target = cfg.processed_dir / cfg.clean_file_name
        repo = CsvPreprocessingRepository(target)
        repo.save(df_clean)
        logger.info(f"Saved cleaned data to: {target}")

        status = 1  # success
    except Exception as e:
        logger.exception(f"Preprocessing failed: {e}")
        raise
    finally:
        duration = time.perf_counter() - t0
        _push_metrics(status=status, duration_s=duration, rows=rows_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run preprocessing")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--params", required=True, help="Path to params.yaml")
    args = parser.parse_args()
    run_preprocessing(args.config, args.params)
