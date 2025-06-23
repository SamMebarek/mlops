from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """
    Configuration pour le module d'entraînement.

    Attributes:
        processed_data_path (Path): Chemin vers le CSV prétraité.
        model_dir (Path): Répertoire où sauvegarder le modèle.
        model_file_name (str): Nom du fichier modèle.
        epochs (int): Nombre d'époques d'entraînement.
        batch_size (int): Taille de batch pour l'entraînement.
        learning_rate (float): Taux d'apprentissage.
        random_seed (int): Seed pour reproductibilité.
        test_size (float): Données réservée pour le test.
    """

    processed_data_path: Path
    model_dir: Path
    model_file_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    random_seed: int
    test_size: float
