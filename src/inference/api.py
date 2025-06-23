from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from starlette.responses import JSONResponse

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


def get_current_admin(
    credentials: HTTPBasicCredentials = Depends(security), request: Request = None
) -> str:
    cfg = request.app.state.cfg
    correct_user = cfg.admin_user
    correct_pass = cfg.admin_password
    if not (
        credentials.username == correct_user and credentials.password == correct_pass
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# --- App initialization ---
app = FastAPI()


@app.on_event("startup")
def init_service():
    cm = ConfigurationManager()
    cfg = cm.get_config()
    # Choose data repository: CSV local first
    data_repo = CsvDataRepository(cfg.data_csv_path)
    # Alternative: instantiate DvcDataRepository
    # data_repo = DvcDataRepository(cfg.dvc_target, cfg.data_csv_path)

    model_repo = MlflowModelRepository(
        tracking_uri=cfg.mlflow_tracking_uri,
        model_name=cfg.mlflow_model_name,
    )

    service = PredictionService(data_repo, model_repo)
    # Store in app.state for access in routes
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
        # attempt load resources
        _ = request.app.state.service.data_repo.load()
        _ = request.app.state.service.model_repo.load()
        return JSONResponse({"status": "OK", "model": "loaded", "data": "loaded"})
    except Exception as e:
        return JSONResponse({"status": "ERROR", "detail": str(e)}, status_code=500)


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest, request: Request):
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


# --- Application entrypoint ---
if __name__ == "__main__":
    cfg = app.state.cfg
    import uvicorn

    uvicorn.run(
        "inference.api:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level.lower(),
    )
