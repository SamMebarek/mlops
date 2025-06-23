# src/inference/utils/common.py
import logging
from pathlib import Path
import yaml
from box import ConfigBox

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    if not path_to_yaml.exists():
        logger.error(f"YAML introuvable : {path_to_yaml}")
        raise FileNotFoundError(f"{path_to_yaml} introuvable")
    with open(path_to_yaml, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    if content is None:
        logger.error(f"YAML vide : {path_to_yaml}")
        raise ValueError(f"{path_to_yaml} est vide")
    logger.info(f"YAML chargé : {path_to_yaml}")
    return ConfigBox(content)


def create_directories(paths: list[Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoire prêt : {p}")
