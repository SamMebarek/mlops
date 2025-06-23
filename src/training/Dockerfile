# Dockerfile pour le service d'entraînement
# Emplacement : src/training/Dockerfile

# 1. Image de base légère Python 3.13.2-slim
FROM python:3.13.2-slim

# 2. Répertoire de travail dans le conteneur
WORKDIR /app

# 3. Copier et installer les dépendances
COPY src/training/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier tout le code du service d'entraînement
COPY src/training/ ./training/

# 5. Variables d'environnement pour logs non bufferisés
ENV PYTHONUNBUFFERED=1

# 6. Point d'entrée : exécution du module training
ENTRYPOINT ["python", "-m", "training.components.train"]
CMD ["--config", "config/config.yaml", "--params", "config/params.yaml"]
