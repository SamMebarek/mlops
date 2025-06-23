import logging
from pathlib import Path
from box import ConfigBox
import yaml
import joblib

# Configuration minimale du logger pour ce module\
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Lit un fichier YAML et renvoie un ConfigBox.

    Args:
        path_to_yaml (Path): Chemin vers le fichier YAML.

    Raises:
        FileNotFoundError: si le fichier n'existe pas.
        ValueError: si le fichier est vide.
    Returns:
        ConfigBox: contenu du YAML convertible en attributs.
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
    Crée une liste de répertoires s'ils n'existent pas.

    Args:
        paths (list[Path]): chemins des répertoires à créer.
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoire créé ou existant : {path}")


def save_model(model: any, path: Path) -> None:
    """
    Sauvegarde un modèle sur disque.

    Args:
        model (any): objet modèle (Scikit-learn, XGBoost, TensorFlow, etc.).
        path (Path): chemin du fichier de sortie (pickle, joblib, h5, etc.).
    """
    create_directories([path.parent])
    # Utilise joblib pour la persistance par défaut
    joblib.dump(model, str(path))
    logger.info(f"Modèle sauvegardé : {path}")
