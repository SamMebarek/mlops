# src/inference/api.py

import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response

from inference.config.configuration import ConfigurationManager
from inference.repository.data_repository import CsvDataRepository
from inference.repository.model_repository import MlflowModelRepository
from inference.service.prediction_service import (
    PredictionService,
    SkuNotFoundError,
    InsufficientDataError,
)
from inference.entity.dto import PredictionResult

# Prometheus
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# --- App initialization ---
app = FastAPI()

# --- Prometheus metrics ---
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency (seconds)",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
INFERENCE_UP = Gauge("inference_up", "Inference up indicator (1=up)")
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total predictions by outcome",
    ["status"],  # success | error
)
PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time spent running model prediction (seconds)",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)

@app.on_event("startup")
def init_service():
    cm = ConfigurationManager()
    cfg = cm.get_config()

    data_repo = CsvDataRepository(cfg.data_csv_path)
    model_repo = MlflowModelRepository(
        tracking_uri=cfg.mlflow_tracking_uri,
        model_name=cfg.mlflow_model_name,
    )
    service = PredictionService(data_repo, model_repo)

    app.state.service = service
    app.state.cfg = cfg
    INFERENCE_UP.set(1)

# --- Middleware for HTTP metrics ---
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    method = request.method
    try:
        resp = await call_next(request)
        status = str(resp.status_code)
        return resp
    except HTTPException as ex:
        status = str(ex.status_code)
        raise
    except Exception:
        status = "500"
        raise
    finally:
        elapsed = time.perf_counter() - start
        # Never let metrics break the request flow
        try:
            HTTP_REQUESTS.labels(method=method, path=path, status=status).inc()
            HTTP_LATENCY.labels(method=method, path=path).observe(elapsed)
        except Exception:
            pass

# --- Pydantic models ---
class PredictionRequest(BaseModel):
    sku: str

class PredictionResponse(BaseModel):
    sku: str
    timestamp: str
    predicted_price: float

# --- Routes ---
@app.get("/health")
def health(request: Request) -> JSONResponse:
    try:
        _ = request.app.state.service.data_repo.load()
        _ = request.app.state.service.model_repo.load()
        return JSONResponse({"status": "OK", "model": "loaded", "data": "loaded"})
    except Exception as e:
        return JSONResponse({"status": "ERROR", "detail": str(e)}, status_code=500)

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, request: Request):
    # Ensure Authorization header present (trust the gateway for JWT/roles)
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    service: PredictionService = request.app.state.service
    t0 = time.perf_counter()
    try:
        result: PredictionResult = service.predict(req.sku)
        PREDICTIONS_TOTAL.labels(status="success").inc()
        PREDICTION_LATENCY.observe(time.perf_counter() - t0)
        return PredictionResponse(
            sku=result.sku,
            timestamp=result.timestamp.isoformat(),
            predicted_price=result.predicted_price,
        )
    except (SkuNotFoundError, InsufficientDataError) as e:
        PREDICTIONS_TOTAL.labels(status="error").inc()
        PREDICTION_LATENCY.observe(time.perf_counter() - t0)
        code = 404 if isinstance(e, SkuNotFoundError) else 400
        raise HTTPException(status_code=code, detail=str(e))
    except Exception as e:
        PREDICTIONS_TOTAL.labels(status="error").inc()
        PREDICTION_LATENCY.observe(time.perf_counter() - t0)
        raise HTTPException(status_code=500, detail="Prediction error: " + str(e))

@app.post("/reload-model")
def reload_model(request: Request):
    # Ensure Authorization header present (trust the gateway for JWT/roles)
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    service: PredictionService = request.app.state.service
    try:
        new_model = service.model_repo.load()
        service.model = new_model
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Reload failed: " + str(e))

# Prometheus scrape endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --- Entrée directe (rarement utilisée) ---
if __name__ == "__main__":
    cfg = app.state.cfg
    import uvicorn
    uvicorn.run(
        "inference.api:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower(),
    )
