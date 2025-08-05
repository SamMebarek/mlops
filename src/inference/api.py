#  src/inference/api.py

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.responses import JSONResponse
import requests

from inference.config.configuration import ConfigurationManager
from inference.repository.data_repository import CsvDataRepository, DvcDataRepository
from inference.repository.model_repository import MlflowModelRepository
from inference.service.prediction_service import (
    PredictionService,
    SkuNotFoundError,
    InsufficientDataError,
)
from inference.entity.dto import PredictionResult

# --- Security setup ---
security = HTTPBasic()
SECURITY_API_URL = "http://security_api:8000/verify-token"

def get_current_admin(
    credentials: HTTPBasicCredentials = Depends(security), request: Request = None
) -> str:
    cfg = request.app.state.cfg
    if credentials.username != cfg.admin_user or credentials.password != cfg.admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

def verify_token_with_security_api(token: str):
    try:
        response = requests.post(SECURITY_API_URL, json={"token": token})
        if response.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid token")
        return response.json()["payload"]
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=500, detail="Token verification service unavailable")

# --- App initialization ---
app = FastAPI()

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
    # 🔐 Vérification du token dans l'en-tête Authorization
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ")[1]
    verify_token_with_security_api(token)

    # ✅ Prédiction si token valide
    service: PredictionService = request.app.state.service
    try:
        result: PredictionResult = service.predict(req.sku)
        return PredictionResponse(
            sku=result.sku,
            timestamp=result.timestamp.isoformat(),
            predicted_price=result.predicted_price,
        )
    except SkuNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except InsufficientDataError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction error: " + str(e))

@app.post("/reload-model", dependencies=[Depends(get_current_admin)])
def reload_model(request: Request):
    service: PredictionService = request.app.state.service
    try:
        new_model = service.model_repo.load()
        service.model = new_model
        return {"message": "Model reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Reload failed: " + str(e))

# --- Entrée directe (rarement utilisé car FastAPI est lancé via uvicorn en ligne de commande) ---
if __name__ == "__main__":
    cfg = app.state.cfg
    import uvicorn
    uvicorn.run(
        "inference.api:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower(),
    )
