# src/inference/config/configuration.py

import os
from pathlib import Path
from box import ConfigBox
from dotenv import load_dotenv

from inference.entity.config_entity import InferenceConfig
from inference.utils.common import read_yaml, create_directories

# Charger automatiquement les variables d'environnement depuis .env à la racine du projet
load_dotenv()


class ConfigurationManager:
    """
    Charge la configuration YAML et les variables d'environnement,
    crée les répertoires nécessaires, et expose un InferenceConfig.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        params_path: str = "config/params.yaml",
    ):
        # Lecture des fichiers YAML
        self.config: ConfigBox = read_yaml(Path(config_path))
        self.params: ConfigBox = read_yaml(Path(params_path))

        # Préparation du répertoire de données
        data_csv = Path(self.config.inference.data_csv_path)
        raw_dir = data_csv.parent
        create_directories([raw_dir])

        # Construction de la configuration d'inférence
        self._inference_config = InferenceConfig(
            data_csv_path=data_csv,
            dvc_target=self.config.inference.dvc_target,
            mlflow_tracking_uri=self.config.inference.mlflow_tracking_uri,
            mlflow_model_name=self.config.inference.mlflow_model_name,
            host=self.config.inference.host,
            port=int(self.config.inference.port),
            log_level=self.config.inference.log_level,
            admin_user=os.getenv("ADMIN_USER", self.config.inference.get("admin_user", None)),
            admin_password=os.getenv(
                "ADMIN_PASSWORD", self.config.inference.get("admin_password", None)
            ),
        )

    def get_config(self) -> InferenceConfig:
        """
        Retourne l'objet InferenceConfig prêt à l'emploi.
        Vérifie la présence des identifiants admin.
        """
        cfg = self._inference_config
        if not cfg.admin_user or not cfg.admin_password:
            raise ValueError("Les variables ADMIN_USER et ADMIN_PASSWORD doivent être définies")
        return cfg
