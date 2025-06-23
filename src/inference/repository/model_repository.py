from abc import ABC, abstractmethod
import mlflow.pyfunc
from typing import Any


class ModelRepository(ABC):
    """
    Interface abstraite pour le chargement des modèles d'inférence.
    Implementations must return a model with a predict() method.
    """

    @abstractmethod
    def load(self) -> Any:
        """Charge et retourne une instance de modèle prête à l'inférence."""
        pass


class MlflowModelRepository(ModelRepository):
    """
    Chargement du modèle depuis le MLflow Model Registry.
    """

    def __init__(
        self, tracking_uri: str, model_name: str, model_stage: str = "Production"
    ):
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.model_stage = model_stage

        #     def load(self) -> Any:
        #         """
        #         Charge la dernière version du modèle dans la stage spécifiée (default: Production).
        #         """
        #         # Construction de l'URI MLflow
        #         model_uri = f"models:/{self.model_name}/{self.model_stage}"
        #         # Configuration de l'URI de tracking
        #         mlflow.pyfunc.tracking_uri = self.tracking_uri
        #         # Chargement du modèle PyFunc
        #         model = mlflow.pyfunc.load_model(model_uri)
        #         return model

        #         --- a/src/inference/repository/model_repository.py
        # +++ b/src/inference/repository/model_repository.py

    def load(self) -> Any:
        #  model_uri = f"models:/{self.model_name}/Production"
        #  return mlflow.pyfunc.load_model(model_uri)
        from mlflow.exceptions import MlflowException

        # Essayer d’abord la version « Production »
        try:
            return mlflow.pyfunc.load_model(f"models:/{self.model_name}/Production")
        except MlflowException:
            # Si elle n’existe pas, basculez sur la dernière dispo
            return mlflow.pyfunc.load_model(f"models:/{self.model_name}/latest")
