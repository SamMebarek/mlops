# services/training/Dockerfile

FROM python:3.10-slim

# Install Git pour MLflow (et clean le cache après)
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Définir le dossier de travail dans le conteneur
WORKDIR /app

# Copier le script et les dépendances
COPY training.py .
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Commande d'exécution par défaut
CMD ["python", "training.py"]
