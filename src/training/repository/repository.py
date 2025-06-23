import logging
from abc import ABC, abstractmethod
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


class ModelRepository(ABC):
    """
    Interface pour la persistance et la récupération des modèles.
    """

    @abstractmethod
    def save(self, model: any) -> None:
        """
        Persiste un modèle entraîné.
        """
        pass

    @abstractmethod
    def load(self) -> any:
        """
        Charge et retourne le modèle enregistré.
        """
        pass


class CsvModelRepository(ModelRepository):
    """
    Implémentation de ModelRepository utilisant joblib et un fichier local.
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path

    def save(self, model: any) -> None:
        """
        Sauvegarde le modèle sur disque.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, str(self.model_path))
        logger.info(f"Modèle sauvegardé → {self.model_path}")

    def load(self) -> any:
        """
        Charge le modèle depuis le disque.
        """
        model = joblib.load(str(self.model_path))
        logger.info(f"Modèle chargé depuis → {self.model_path}")
        return model
