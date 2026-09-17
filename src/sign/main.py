"""ASGI entry point: ``uvicorn src.sign.main:app``."""

from .api import create_app

app = create_app()
