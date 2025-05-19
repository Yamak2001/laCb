# backend/main.py
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import config

# Initialize directories
from init_directories import initialize_directories
initialize_directories()

# Import model factories to register models
from models.factory import register_all_models

# Import your routers
from routers import voice_profiles, audio_processing

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize models
logger.info("Registering available models")
register_all_models()
logger.info("Model registration complete")

app = FastAPI(
    title="Loud & Clear API",
    description="Voice isolation system with modular model selection",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.get_config("CORS_ORIGIN", "http://localhost:5173")],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # Create a monitoring instance
# request_monitor = get_request_monitor(store_limit=200)

# # Add monitoring middleware with the monitor instance
# app.add_middleware(MonitoringMiddleware)

# # Store the monitor in app.state for access in endpoints
# app.state.request_monitor = request_monitor

# Include routers
app.include_router(voice_profiles.router)
app.include_router(audio_processing.router)
# app.include_router(monitoring.router)

# Mount static files
app.mount("/data/audio", StaticFiles(directory="data/audio"), name="audio")
app.mount("/data/isolated", StaticFiles(directory="data/isolated"), name="isolated")
app.mount("/data/separated", StaticFiles(directory="data/separated"), name="separated")
app.mount("/data/mixed", StaticFiles(directory="data/mixed"), name="mixed")

@app.get("/")
async def root():
    return {
        "message": "Loud & Clear API is running",
        "version": "2.0.0",
        "description": "Voice isolation system with modular model selection"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    from models.factory import SeparationModelFactory, EmbeddingModelFactory, VADProcessorFactory
    
    return {
        "status": "healthy",
        "available_models": {
            "separation": list(SeparationModelFactory.list_available_models().keys()),
            "embedding": list(EmbeddingModelFactory.list_available_models().keys()),
            "vad": list(VADProcessorFactory.list_available_processors().keys())
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to log all exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.API_HOST, port=config.API_PORT, reload=True)