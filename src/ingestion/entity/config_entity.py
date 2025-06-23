# src/ingestion/entity/config_entity.py

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Contient les paramètres d’ingestion lus depuis config.yaml :
      - source_URL : URL ou chemin local du CSV à ingérer
      - raw_data_dir : dossier où stocker le CSV ingéré
      - ingested_file_name : nom du fichier ingéré
    """

    source_URL: str
    raw_data_dir: Path
    ingested_file_name: str
