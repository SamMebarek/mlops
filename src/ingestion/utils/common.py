# src/ingestion/config/configuration.py

import logging
from pathlib import Path
from box import ConfigBox
import yaml

# Configuration minimale du logger pour ce module
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Lit un fichier YAML et renvoie un ConfigBox.
    """
    if not path_to_yaml.exists():
        logger.error(f"Fichier YAML introuvable : {path_to_yaml}")
        raise FileNotFoundError(f"{path_to_yaml} introuvable")
    with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
        content = yaml.safe_load(yaml_file)
        if content is None:
            logger.error(f"Fichier YAML vide : {path_to_yaml}")
            raise ValueError(f"{path_to_yaml} est vide")
        logger.info(f"YAML chargé : {path_to_yaml}")
        return ConfigBox(content)


def create_directories(paths: list[Path]):
    """
    Crée une liste de répertoires si absents.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoire créé ou existant : {path}")
