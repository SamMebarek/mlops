from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PredictionResult:
    """
    Data Transfer Object pour encapsuler le résultat d'une prédiction.

    Attributes:
        sku: str               # Identifiant du produit prédit
        timestamp: datetime    # Horodatage de la prédiction
        predicted_price: float # Prix prédit (arrondi à 2 décimales)
    """

    sku: str
    timestamp: datetime
    predicted_price: float
