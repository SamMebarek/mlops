FROM python:3.10-slim

# Installer Git si nécessaire (ex : pour MLflow ou logs Git, optionnel ici)
RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY preprocessing.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "preprocessing.py"]
