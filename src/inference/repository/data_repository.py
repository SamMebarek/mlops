#  src/inference/repository/data_repository.py

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import pandas as pd


class DataRepository(ABC):
    """
    Interface abstraite pour le chargement des données d'inférence.
    Implementations must return a pandas DataFrame.
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Charge et retourne un DataFrame contenant les données."""
        pass


class CsvDataRepository(DataRepository):
    """
    Chargement des données depuis un fichier CSV local.
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding="utf-8")
        return df


class DvcDataRepository(DataRepository):
    """
    Chargement des données versionnées via DVC.

    This implementation pulls the latest data from DVC remote before loading locally.
    """

    def __init__(self, dvc_target: str, csv_path: Path):
        # dvc_target: the path in DVC (e.g. "data/raw/ingested_data.csv")
        self.dvc_target = dvc_target
        self.csv_path = csv_path

    def load(self) -> pd.DataFrame:
        # Ensure latest from DVC
        try:
            subprocess.run(["dvc", "pull", self.dvc_target], check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"DVC pull failed for {self.dvc_target}: {e}")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found after DVC pull: {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding="utf-8")
        return df
