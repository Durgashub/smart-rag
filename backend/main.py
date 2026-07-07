"""
main.py — FastAPI application entry point.

Creates the app, registers CORS middleware, and includes all routers.
This file should stay small (~30 lines). Business logic lives in services/.
Routing logic lives in routers/. Everything else is in config.py.

Run with:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import ALLOWED_ORIGINS, ALLOWED_ORIGIN_REGEX
from routers import system, files, ask, career, suggestions

app = FastAPI(title="SmartRAG AI — Stage 4 (Modular)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "x-session-id"],
    expose_headers=["x-session-id"],
)

app.include_router(system.router)
app.include_router(files.router)
app.include_router(ask.router)
app.include_router(career.router)
app.include_router(suggestions.router)
