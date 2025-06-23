# Dockerfile pour le service de preprocessing
# Emplacement : src/preprocessing/Dockerfile

# 1. Image de base légère correspondant à Python 3.13.2
FROM python:3.13.2-slim

# 2. Répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier et installer les dépendances
COPY src/preprocessing/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier tout le code du service de preprocessing
COPY src/preprocessing/ ./preprocessing/

# 5. Variables d'environnement pour logs non bufferisés
ENV PYTHONUNBUFFERED=1

# 6. Point d'entrée : exécution du module preprocess
ENTRYPOINT ["python", "-m", "preprocessing.components.preprocess"]
CMD ["--config", "config/config.yaml", "--params", "config/params.yaml"]
