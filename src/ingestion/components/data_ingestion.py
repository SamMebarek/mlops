# src/ingestion/components/data_ingestion.py

import argparse
import logging
import hashlib
import os
import time
from pathlib import Path

import pandas as pd

from ingestion.config.configuration import ConfigurationManager
from ingestion.repository.repository import CsvIngestionRepository

# Prometheus Pushgateway (minimal, safe)
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)

PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
PUSH_GROUPING = {"instance": os.getenv("INSTANCE", "local"), "service": "ingestion"}


def _push_metrics(status: int, duration_s: float, rows: int, out_bytes: int) -> None:
    """Push a few gauges; never raise."""
    try:
        reg = CollectorRegistry()
        g_status = Gauge("job_status", "1=success,0=failure", registry=reg)
        g_dur = Gauge("job_duration_seconds", "Job wall time in seconds", registry=reg)
        g_rows = Gauge("rows_processed_total", "Rows processed by the job", registry=reg)
        g_size = Gauge(
            "output_file_size_bytes",
            "Size of the written CSV (bytes)",
            registry=reg,
        )

        g_status.set(float(status))
        g_dur.set(float(duration_s))
        g_rows.set(float(rows))
        g_size.set(float(out_bytes))

        push_to_gateway(
            PUSHGATEWAY_URL,
            job="ingestion",
            grouping_key=PUSH_GROUPING,
            registry=reg,
        )
        logger.info(
            f"Pushed metrics to Pushgateway ({PUSHGATEWAY_URL}) "
            f"status={status} duration_s={duration_s:.3f} rows={rows} out_bytes={out_bytes}"
        )
    except Exception as e:
        logger.warning(f"Failed to push metrics to Pushgateway: {e}")


def calculate_md5(file_path: Path) -> str:
    """Calcule le hash MD5 en lisant par blocs pour gérer les gros fichiers."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_schema(df: pd.DataFrame):
    """
    Vérifie que le DataFrame n’est pas vide, que 'SKU' et 'Prix' sont présents,
    et que 'Prix' est de type numérique.
    """
    if df.empty:
        raise ValueError("Le DataFrame est vide.")
    required_columns = ["SKU", "Prix"]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    if not pd.api.types.is_numeric_dtype(df["Prix"]):
        raise ValueError("La colonne 'Prix' doit être numérique.")


def run_ingestion(config_path: str, params_path: str):
    """
    1. Charge config et params.
    2. Lit le CSV (URL ou local).
    3. Valide le schéma minimal.
    4. Persiste via CsvIngestionRepository.
    """
    t0 = time.perf_counter()
    status = 0
    rows_read = 0
    out_bytes = 0

    cm = ConfigurationManager(config_path, params_path)
    ingestion_cfg = cm.get_data_ingestion_config()
    params = cm.get_params()

    source_url = ingestion_cfg.source_URL
    raw_dir = ingestion_cfg.raw_data_dir
    ingested_file = raw_dir / ingestion_cfg.ingested_file_name

    try:
        # Lecture du CSV
        if source_url.startswith(("http://", "https://")):
            df = pd.read_csv(source_url, sep=",", encoding="utf-8", low_memory=False)
            logger.info(f"CSV téléchargé depuis URL : {source_url}")
        else:
            df = pd.read_csv(source_url, sep=",", encoding="utf-8", low_memory=False)
            logger.info(f"CSV lu localement : {source_url}")
        rows_read = len(df)

        # Validation minimale
        validate_schema(df)
        logger.info("Validation du schéma réussie.")

        # Persistance via repository CSV
        repo = CsvIngestionRepository(ingested_file)
        repo.save(df)
        if ingested_file.exists():
            out_bytes = ingested_file.stat().st_size
        md5 = calculate_md5(ingested_file)
        logger.info(f"Fichier ingéré (CSV) sauvegardé : {ingested_file} (MD5={md5})")

        status = 1  # success
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        duration = time.perf_counter() - t0
        _push_metrics(status=status, duration_s=duration, rows=rows_read, out_bytes=out_bytes)


if __name__ == "__main__":
    # Configuration du logger pour le CLI (stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingestion de données CSV")
    parser.add_argument("--config", required=True, help="Chemin vers config/config.yaml")
    parser.add_argument("--params", required=True, help="Chemin vers config/params.yaml")
    args = parser.parse_args()
    run_ingestion(args.config, args.params)
