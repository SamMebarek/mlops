# Dockerfile pour le service d'inférence
# Emplacement : src/inference/Dockerfile

# 1. Image de base légère Python 3.13.2
FROM python:3.13.2-slim

# 2. Répertoire de travail
WORKDIR /app

# 3. Copier et installer les dépendances
COPY src/inference/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier tout le code du service d'inférence
COPY src/inference/ ./inference/

# 5. Copier la configuration et les éventuels fichiers .env
COPY config/ ./config/

# 6. Variables pour logs non bufferisés
ENV PYTHONUNBUFFERED=1

# 7. Exposer le port défini dans la config (par défaut 8080)
EXPOSE 8080

# 8. Point d'entrée : lancement Uvicorn
ENTRYPOINT ["uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
