# backend/main.py
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import your routers
from routers import voice_profiles, audio_processing, monitoring

# Import your middleware
from middleware.monitoring import MonitoringMiddleware

app = FastAPI(title="Loud & Clear API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware - single step implementation
monitoring_middleware = MonitoringMiddleware(app, store_limit=200)
# Store middleware instance for access in endpoints
app.state.monitoring_middleware = monitoring_middleware

os.makedirs("data/audio", exist_ok=True)
os.makedirs("data/isolated", exist_ok=True)
os.makedirs("data/separated", exist_ok=True)
os.makedirs("data/temp", exist_ok=True)

# Include routers
app.include_router(voice_profiles.router)
app.include_router(audio_processing.router)
app.include_router(monitoring.router)

# Mount static files
app.mount("/data/audio", StaticFiles(directory="data/audio"), name="audio")
app.mount("/data/isolated", StaticFiles(directory="data/isolated"), name="isolated")
app.mount("/data/separated", StaticFiles(directory="data/separated"), name="separated")
@app.get("/")
async def root():
    return {"message": "Loud & Clear API is running"}
