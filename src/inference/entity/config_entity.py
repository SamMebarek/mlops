from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    """
    Configuration pour le service d'inférence.

    Attributes:
        data_csv_path: Path          # Chemin local vers le CSV de données
        dvc_target: str              # Cible DVC (ex: "data/raw/ingested_data.csv")
        mlflow_tracking_uri: str     # URI du serveur MLflow (ex: DagsHub)
        mlflow_model_name: str       # Nom du modèle dans le registry MLflow
        host: str                    # Adresse d'écoute de FastAPI (ex: "0.0.0.0")
        port: int                    # Port d'écoute (ex: 8080)
        log_level: str               # Niveau de log (ex: "INFO")
        admin_user: str              # Utilisateur HTTP Basic pour endpoints admin
        admin_password: str          # Mot de passe HTTP Basic pour endpoints admin
    """

    data_csv_path: Path
    dvc_target: str
    mlflow_tracking_uri: str
    mlflow_model_name: str
    host: str
    port: int
    log_level: str
    admin_user: str
    admin_password: str
