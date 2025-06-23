from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration pour le module d'évaluation.

    Attributes:
        processed_data_path (Path): Chemin vers le CSV prétraité.
        model_path (Path): Chemin vers le modèle entraîné.
        metrics_output_path (Path): Chemin de sortie pour les métriques JSON.
        mlflow_tracking_uri (str): URI du serveur MLflow.
        mlflow_experiment_name (str): Nom de l'expérience MLflow.
        mlflow_model_name (str): Nom sous lequel enregistrer le modèle.
    """

    processed_data_path: Path
    model_path: Path
    metrics_output_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_model_name: str
