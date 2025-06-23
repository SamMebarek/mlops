import logging
from pathlib import Path
from box import ConfigBox
import yaml
import json

# Logger configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Lit un fichier YAML et renvoie un ConfigBox.

    Args:
        path_to_yaml (Path): chemin vers le fichier YAML.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si le fichier est vide.

    Returns:
        ConfigBox: contenu du YAML.
    """
    if not path_to_yaml.exists():
        logger.error(f"Fichier YAML introuvable : {path_to_yaml}")
        raise FileNotFoundError(f"{path_to_yaml} introuvable")
    with open(path_to_yaml, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
        if content is None:
            logger.error(f"Fichier YAML vide : {path_to_yaml}")
            raise ValueError(f"{path_to_yaml} est vide")
        logger.info(f"YAML chargé : {path_to_yaml}")
        return ConfigBox(content)


def create_directories(paths: list[Path]) -> None:
    """
    Crée récursivement une liste de répertoires.

    Args:
        paths (list[Path]): liste de chemins de répertoires.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoire créé ou existant : {path}")


def save_json(path: Path, data: dict) -> None:
    """
    Sauvegarde un dictionnaire au format JSON.

    Args:
        path (Path): chemin du fichier de sortie.
        data (dict): données à sauvegarder.
    """
    create_directories([path.parent])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logger.info(f"JSON sauvegardé : {path}")


def load_json(path: Path) -> dict:
    """
    Charge un fichier JSON et renvoie son contenu.

    Args:
        path (Path): chemin du fichier JSON.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.

    Returns:
        dict: contenu du JSON.
    """
    if not path.exists():
        logger.error(f"Fichier JSON introuvable : {path}")
        raise FileNotFoundError(f"{path} introuvable")
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    logger.info(f"JSON chargé : {path}")
    return content
