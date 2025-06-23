from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configuration pour le module de prétraitement.

    Attributes:
        raw_data_path (Path): Chemin vers le fichier ingéré (data/raw/ingested_data.csv).
        processed_dir (Path): Répertoire où sauvegarder les données prétraitées (data/processed).
        clean_file_name (str): Nom du fichier prétraité (clean_data.csv).
    """

    raw_data_path: Path
    processed_dir: Path
    clean_file_name: str
