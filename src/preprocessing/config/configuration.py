import logging
from pathlib import Path
from box import ConfigBox

from preprocessing.utils.common import read_yaml, create_directories
from preprocessing.entity.config_entity import PreprocessingConfig

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """
    Charge les fichiers YAML de config et params, crée les répertoires nécessaires,
    et expose un PreprocessingConfig pour le module de prétraitement.
    """

    def __init__(self, config_filepath: str, params_filepath: str):
        config_path = Path(config_filepath).resolve()
        project_root = config_path.parent.parent  # remonte deux niveaux : src/ → project_root/
        self.config = read_yaml(config_path)
        self.params = read_yaml(Path(params_filepath))
        # Crée directement data/processed sous la racine du projet
        processed_dir = project_root / self.config.data_preprocessing.processed_dir
        create_directories([processed_dir])
        self._project_root = project_root

    def get_preprocessing_config(self) -> PreprocessingConfig:
        cfg = self.config.data_preprocessing
        raw_data_rel = Path(cfg.raw_data_path)
        raw_data_path = self._project_root / raw_data_rel
        processed_dir = self._project_root / Path(cfg.processed_dir)
        create_directories([processed_dir])
        return PreprocessingConfig(
            raw_data_path=raw_data_path,
            processed_dir=processed_dir,
            clean_file_name=cfg.clean_file_name,
        )

    def get_params(self) -> ConfigBox:
        return self.params.preprocessing
