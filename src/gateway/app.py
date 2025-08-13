# src/gateway/app.py

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
import jwt
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# --- Prometheus ---
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response

# --- Config ---
JWT_SECRET = os.getenv("JWT_SECRET", "jwtsecret")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 120

USER_CREDENTIALS = {
    os.getenv("ADMIN_USER", "admin"): "admin",  # default: admin/admin
    os.getenv("NORMAL_USER", "user"): "user",  # default: user/user
}
ROLES = {
    os.getenv("ADMIN_USER", "admin"): "admin",
    os.getenv("NORMAL_USER", "user"): "user",
}

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference:8080/predict")
RELOAD_URL = os.getenv("INFERENCE_RELOAD_URL", "http://inference:8080/reload-model")


# --- Models ---
class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class PredictionRequest(BaseModel):
    sku: str


class PredictionResponse(BaseModel):
    sku: str
    timestamp: str
    predicted_price: float


# --- App ---
app = FastAPI()

# --- Prometheus metrics (low-cardinality labels only) ---
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

GATEWAY_UP = Gauge("gateway_up", "Gateway up indicator (1=up)")


@app.on_event("startup")
def mark_up():
    GATEWAY_UP.set(1)


# Middleware to record request count & latency
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()

    # Use templated route path when available to avoid high cardinality
    path = request.scope.get("route").path if request.scope.get("route") else request.url.path
    method = request.method

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except HTTPException as ex:
        status = str(ex.status_code)
        raise
    except Exception:
        status = "500"
        raise
    finally:
        elapsed = time.perf_counter() - start
        # Record metrics
        try:
            HTTP_REQUESTS.labels(method=method, path=path, status=status).inc()
            HTTP_LATENCY.labels(method=method, path=path).observe(elapsed)
        except Exception:
            # Never let metrics break the request path
            pass


# --- Utility functions ---
def create_jwt(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user_role(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ")[1]
    payload = decode_jwt(token)
    return payload.get("role", "")


# --- Routes ---
@app.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    username, password = req.username, req.password
    if username not in USER_CREDENTIALS or USER_CREDENTIALS[username] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    role = ROLES[username]
    token = create_jwt(username, role)
    return LoginResponse(access_token=token, role=role)


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest, request: Request):
    role = get_current_user_role(request)
    if role not in ("user", "admin"):
        raise HTTPException(status_code=403, detail="Forbidden")
    token = request.headers.get("Authorization")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                INFERENCE_URL,
                headers={"Authorization": token},
                json=req.dict(),
                timeout=10.0,
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Inference service unavailable")


@app.post("/reload-model")
async def reload_model(request: Request):
    role = get_current_user_role(request)
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    token = request.headers.get("Authorization")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                RELOAD_URL,
                headers={"Authorization": token},
                timeout=10.0,
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
        except httpx.RequestError:
            raise HTTPException(status_code=503, detail="Inference service unavailable")


@app.get("/health")
def health():
    return {"status": "OK", "service": "gateway"}


# Prometheus scrape endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gateway.app:app", host="0.0.0.0", port=8002, reload=True)
