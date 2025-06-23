# Dockerfile for frontend service (FastAPI + Jinja2)
# Location: src/frontend/Dockerfile

# 1. Base image
FROM python:3.13.2-slim

# 2. Set working directory
WORKDIR /app

# 3. Install dependencies
# Copy requirements (assumes a requirements.txt in src/frontend)
COPY src/frontend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy application code
COPY src/frontend/ ./frontend/

COPY ../../config /app/config

# 5. Expose port
EXPOSE 8000

# 6. Environment variables
ENV PYTHONUNBUFFERED=1

# 7. Entrypoint
CMD ["uvicorn", "frontend.app:app", "--host", "0.0.0.0", "--port", "8000"]
# 