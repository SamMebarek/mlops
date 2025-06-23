# src/ingestion/config/configuration.py

import logging
from pathlib import Path
from box import ConfigBox

from ingestion.utils.common import read_yaml, create_directories
from ingestion.entity.config_entity import DataIngestionConfig

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Charge les fichiers YAML de config et params, crée les répertoires nécessaires,
    et expose un DataIngestionConfig pour le module d’ingestion.
    """

    def __init__(self, config_filepath: str, params_filepath: str):
        # Charger le contenu des deux YAML
        self.config = read_yaml(Path(config_filepath))
        self.params = read_yaml(Path(params_filepath))
        # Créer le dossier parent de raw_data_dir si besoin
        raw_dir_parent = Path(self.config.data_ingestion.raw_data_dir).parent
        create_directories([raw_dir_parent])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config.data_ingestion
        raw_dir = Path(cfg.raw_data_dir)
        # Veiller à ce que le dossier raw existe
        create_directories([raw_dir])
        return DataIngestionConfig(
            source_URL=cfg.source_URL,
            raw_data_dir=raw_dir,
            ingested_file_name=cfg.ingested_file_name,
        )

    def get_params(self) -> ConfigBox:
        return self.params.ingestion
