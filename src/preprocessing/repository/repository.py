import logging
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class PreprocessingRepository(ABC):
    """
    Interface pour la persistance des données prétraitées.
    """

    @abstractmethod
    def save(self, df: pd.DataFrame) -> None:
        """
        Persiste le DataFrame transformé.
        """
        pass


class CsvPreprocessingRepository(PreprocessingRepository):
    """
    Implémentation CSV du repository de prétraitement.
    """

    def __init__(self, target_path: Path):
        self.target_path = target_path

    def save(self, df: pd.DataFrame) -> None:
        """
        Sauvegarde le DataFrame en CSV, crée le dossier si nécessaire.
        """
        # Créer le dossier parent si besoin
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.target_path, index=False, encoding="utf-8")
        logger.info(f"Persistance CSV nettoyé → {self.target_path}")
