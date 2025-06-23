from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
import yaml
from pathlib import Path

# --- App initialization ---
app = FastAPI()

# BASE_DIR doit pointer sur le dossier src/frontend
BASE_DIR = Path(__file__).parent  # src/frontend
# Pour accéder à la config située à la racine du projet dans /config
CONFIG_DIR = BASE_DIR.parent.parent / "config"  # src/frontend → src → racine/config

# Mount static assets (même si vide) depuis src/frontend/static
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Template engine pointant sur src/frontend/templates
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Load frontend config (contient notamment l'URL de l'API)
with open(CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
    frontend_cfg = yaml.safe_load(f)
API_URL = frontend_cfg.get("api_url", "http://localhost:8080")


# --- Routes for each microservice page ---
@app.get("/ingestion", response_class=HTMLResponse)
async def ingestion_page(request: Request):
    return templates.TemplateResponse(
        "ingestion.html",
        {"request": request, "api_url": API_URL},
    )


@app.get("/preprocessing", response_class=HTMLResponse)
async def preprocessing_page(request: Request):
    return templates.TemplateResponse(
        "preprocessing.html",
        {"request": request, "api_url": API_URL},
    )


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    return templates.TemplateResponse(
        "training.html",
        {"request": request, "api_url": API_URL},
    )


@app.get("/evaluation", response_class=HTMLResponse)
async def evaluation_page(request: Request):
    return templates.TemplateResponse(
        "evaluation.html",
        {"request": request, "api_url": API_URL},
    )


@app.get("/inference", response_class=HTMLResponse)
async def inference_page(request: Request):
    return templates.TemplateResponse(
        "inference.html",
        {"request": request, "api_url": API_URL},
    )


# Health check for frontend
@app.get("/health")
def health():
    return {"status": "OK", "frontend": "running"}


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse("/ingestion")
