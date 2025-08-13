import logging
from pathlib import Path
from evaluation.entity.config_entity import EvaluationConfig
from evaluation.utils.common import read_yaml, create_directories

# Logger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Charge les fichiers YAML de config et params pour le module d'évaluation,
    crée les répertoires nécessaires et expose une EvaluationConfig.
    """

    def __init__(self, config_path: str, params_path: str):
        # Chargement des YAML
        self.config = read_yaml(Path(config_path))
        self.params = read_yaml(Path(params_path))

        # Détermination de la racine du projet (2 niveaux au-dessus de src/evaluation/config)
        self.project_root = Path(config_path).resolve().parent.parent

        # Création du dossier pour les métriques
        metrics_dir = (
            self.project_root / Path(self.config.evaluation.metrics_output_path).parent
        )
        create_directories([metrics_dir])

    def get_evaluation_config(self) -> EvaluationConfig:
        cfg = self.config.evaluation
        # Chemins absolus
        processed_data = self.project_root / Path(cfg.processed_data_path)
        model_file = self.project_root / Path(cfg.model_path)
        metrics_output = self.project_root / Path(cfg.metrics_output_path)

        return EvaluationConfig(
            processed_data_path=processed_data,
            model_path=model_file,
            metrics_output_path=metrics_output,
            mlflow_tracking_uri=cfg.mlflow_tracking_uri,
            mlflow_experiment_name=cfg.mlflow_experiment_name,
            mlflow_model_name=cfg.mlflow_model_name,
        )
