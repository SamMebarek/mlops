# src/inference/service/prediction_service.py

from typing import Any, List
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from inference.repository.data_repository import DataRepository
from inference.repository.model_repository import ModelRepository
from inference.entity.dto import PredictionResult


class SkuNotFoundError(Exception):
    """Levée lorsque le SKU demandé n'existe pas dans les données."""

    pass


class InsufficientDataError(Exception):
    """Levée lorsqu'il n'y a pas assez d'observations pour prédire."""

    pass


class PredictionService:
    """
    Service métier pour réaliser l'inférence de prix.
    """

    def __init__(self, data_repo: DataRepository, model_repo: ModelRepository):
        # Injection des repositories
        self.data_repo = data_repo
        self.model_repo = model_repo  # <-- On garde la référence au repository
        # Chargement initial du modèle
        self.model = model_repo.load()

    def predict(self, sku: str) -> PredictionResult:
        # 1. Charger le DataFrame complet
        df = self.data_repo.load()

        # 2. Filtrer pour le SKU
        sku_df = df[df["SKU"] == sku]
        if sku_df.empty:
            raise SkuNotFoundError(f"SKU '{sku}' non trouvé dans les données.")

        # 3. Prendre les 3 dernières observations
        last_obs = sku_df.sort_values(by="Timestamp", ascending=False).head(3)
        if len(last_obs) < 3:
            raise InsufficientDataError(
                f"Données insuffisantes pour le SKU '{sku}': {len(last_obs)} enregistrements."
            )

        # 4. Moyenne des features numériques
        numeric_features: List[str] = [
            "PrixInitial",
            "AgeProduitEnJours",
            "QuantiteVendue",
            "UtiliteProduit",
            "ElasticitePrix",
            "Remise",
            "Qualite",
        ]
        mean_vals = last_obs[numeric_features].mean().to_dict()

        # 5. Ajout des features temporelles
        now = datetime.now()
        mean_vals["Mois_sin"] = np.sin(2 * np.pi * now.month / 12)
        mean_vals["Mois_cos"] = np.cos(2 * np.pi * now.month / 12)
        mean_vals["Heure_sin"] = np.sin(2 * np.pi * now.hour / 24)
        mean_vals["Heure_cos"] = np.cos(2 * np.pi * now.hour / 24)

        # 6. Construire le DataFrame d'une ligne
        feature_order = numeric_features + [
            "Mois_sin",
            "Mois_cos",
            "Heure_sin",
            "Heure_cos",
        ]
        feature_row = pd.DataFrame([mean_vals])[feature_order]

        # 7. Prédiction
        predicted_array = self.model.predict(feature_row)
        predicted_price = float(predicted_array[0])

        # 8. Retourner le DTO
        return PredictionResult(sku=sku, timestamp=now, predicted_price=round(predicted_price, 2))
