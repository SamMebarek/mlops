# 1. Image de base
FROM python:3.13.2-slim

# 2. Répertoire de travail
WORKDIR /app

# 3. Copier et installer les dépendances
COPY src/evaluation/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code d’évaluation
COPY src/evaluation/ ./evaluation/

# 5. Logs non bufferisés
ENV PYTHONUNBUFFERED=1

# 6. Entrypoint et arguments par défaut
ENTRYPOINT ["python", "-m", "evaluation.components.evaluate"]
CMD ["--config", "config/config.yaml", "--params", "config/params.yaml"]
