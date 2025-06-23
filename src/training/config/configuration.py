import logging
from pathlib import Path
from box import ConfigBox

from training.utils.common import read_yaml, create_directories
from training.entity.config_entity import TrainingConfig

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Charge les fichiers YAML de config et params, crée les répertoires nécessaires,
    et expose un TrainingConfig pour le module d'entraînement.
    """

    def __init__(self, config_filepath: str, params_filepath: str):
        # Résolution des chemins
        config_path = Path(config_filepath).resolve()
        params_path = Path(params_filepath).resolve()

        # Chargement des YAML
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        # Racine du projet (deux niveaux au-dessus de config/)
        project_root = config_path.parent.parent

        # Création du dossier de sortie du modèle
        model_dir = project_root / self.config.training.model_dir
        create_directories([model_dir])

        self._project_root = project_root

    def get_training_config(self) -> TrainingConfig:
        """
        Extrait et retourne la configuration d'entraînement.
        """
        cfg = self.config.training
        p = self.params.training

        # Chemin absolu vers les données traitées
        processed_data_path = self._project_root / Path(cfg.processed_data_path)
        # Répertoire de sortie pour le modèle
        model_dir = self._project_root / Path(cfg.model_dir)

        # S'assurer que le dossier existe
        create_directories([model_dir])

        return TrainingConfig(
            processed_data_path=processed_data_path,
            model_dir=model_dir,
            model_file_name=cfg.model_file_name,
            epochs=p.epochs,
            batch_size=p.batch_size,
            learning_rate=p.learning_rate,
            random_seed=p.random_seed,
            test_size=p.test_size,
        )

    def get_model_config(self) -> ConfigBox:
        """
        Retourne la section 'model_config' (MLflow, etc.)
        """
        return self.config.model_config
