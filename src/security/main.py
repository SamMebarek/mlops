#  src/security/main.py

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import jwt
from datetime import datetime, timedelta

app = FastAPI()

@app.post("/token")
def login(username: str = Form(...), password: str = Form(...)):
    admin_user = os.getenv("ADMIN_USER")
    admin_password = os.getenv("ADMIN_PASSWORD")
    secret_key = os.getenv("JWT_SECRET", "supersecretkey123")

    if username != admin_user or password != admin_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = jwt.encode(
        {"sub": username, "exp": datetime.utcnow() + timedelta(hours=1)},
        secret_key,
        algorithm="HS256",
    )

    return JSONResponse(content={"access_token": token, "token_type": "bearer"})
