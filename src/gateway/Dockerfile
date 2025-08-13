# Dockerfile for the Gateway Service
# Location: src/gateway/Dockerfile

# 1. Use a slim Python image (match version to your project, e.g. 3.13)
FROM python:3.13.2-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy and install dependencies
COPY src/gateway/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy all the gateway code
COPY src/gateway/ ./gateway/

# 5. Copy .env and (optionally) config files if needed

# 6. Set unbuffered logs (optional, good for Docker)
ENV PYTHONUNBUFFERED=1

# 7. Expose the gateway port (default 8002)
EXPOSE 8002

# 8. Entrypoint: launch Uvicorn (assumes main file is gateway/app.py)
ENTRYPOINT ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8002"]
