# src/ingestion/repository/repository.py

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class IngestionRepository(ABC):
    """
    Interface pour la persistance des données ingérées.
    """

    @abstractmethod
    def save(self, df: pd.DataFrame) -> None:
        pass


class CsvIngestionRepository(IngestionRepository):
    """
    Implémentation CSV de IngestionRepository.
    """

    def __init__(self, target_path: Path):
        self.target_path = target_path

    def save(self, df: pd.DataFrame) -> None:
        df.to_csv(self.target_path, index=False, encoding="utf-8")
        logger.info(f"Persistance CSV effectuée → {self.target_path}")
