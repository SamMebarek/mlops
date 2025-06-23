# src/ingestion/components/data_ingestion.py

import argparse
import logging
import hashlib
import pandas as pd
from pathlib import Path

from ingestion.config.configuration import ConfigurationManager
from ingestion.repository.repository import CsvIngestionRepository

logger = logging.getLogger(__name__)


def calculate_md5(file_path: Path) -> str:
    """
    Calcule le hash MD5 en lisant par blocs pour gérer les gros fichiers.
    """
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
    cm = ConfigurationManager(config_path, params_path)
    ingestion_cfg = cm.get_data_ingestion_config()
    params = cm.get_params()

    source_url = ingestion_cfg.source_URL
    raw_dir = ingestion_cfg.raw_data_dir
    ingested_file = raw_dir / ingestion_cfg.ingested_file_name

    # Lecture du CSV
    try:
        if source_url.startswith(("http://", "https://")):
            df = pd.read_csv(source_url, sep=",", encoding="utf-8", low_memory=False)
            logger.info(f"CSV téléchargé depuis URL : {source_url}")
        else:
            df = pd.read_csv(source_url, sep=",", encoding="utf-8", low_memory=False)
            logger.info(f"CSV lu localement : {source_url}")
    except Exception as e:
        logger.error(f"Erreur de lecture du CSV : {e}")
        raise

    # Validation minimale
    try:
        validate_schema(df)
        logger.info("Validation du schéma réussie.")
    except Exception as e:
        logger.error(f"Échec de la validation du schéma : {e}")
        raise

    # Persistance via repository CSV
    try:
        repo = CsvIngestionRepository(ingested_file)
        repo.save(df)
        md5 = calculate_md5(ingested_file)
        logger.info(f"Fichier ingéré (CSV) sauvegardé : {ingested_file} (MD5={md5})")
    except Exception as e:
        logger.error(f"Échec de la persistance CSV : {e}")
        raise


if __name__ == "__main__":
    # Configuration du logger pour le CLI (stdout)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Ingestion de données CSV")
    parser.add_argument(
        "--config", required=True, help="Chemin vers config/config.yaml"
    )
    parser.add_argument(
        "--params", required=True, help="Chemin vers config/params.yaml"
    )
    args = parser.parse_args()
    run_ingestion(args.config, args.params)
