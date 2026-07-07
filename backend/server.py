# server.py — backward-compatibility shim for Railway deployment.
# Railway's startCommand is: uvicorn server:app ...
# This file just re-exports `app` from main.py so no Railway config change is needed.
from main import app  # noqa: F401
