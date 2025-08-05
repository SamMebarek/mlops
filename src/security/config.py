# src/security/config.py

import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    jwt_secret: str = os.getenv("JWT_SECRET", "defaultsecretkey")
    admin_user: str = os.getenv("ADMIN_USER", "admin")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "password")

settings = Settings()
