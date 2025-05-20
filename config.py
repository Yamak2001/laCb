"""
Configuration settings for Loud & Clear voice isolation system
"""

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Data subdirectories
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
ISOLATED_DIR = os.path.join(DATA_DIR, "isolated")
MIXED_DIR = os.path.join(DATA_DIR, "mixed")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

# Model directories
MODELS_DIR = os.path.join(BASE_DIR, "models")
PRETRAINED_DIR = os.path.join(MODELS_DIR, "pretrained")

# Audio settings
SAMPLE_RATE = 16000
MIN_DURATION = 3.0  # Minimum audio duration in seconds
MAX_DURATION = 10.0  # Maximum audio duration in seconds

# Model defaults
DEFAULT_SEPARATION_MODEL = "convtasnet"
DEFAULT_EMBEDDING_MODEL = "resemblyzer"
DEFAULT_VAD_PROCESSOR = "webrtcvad"

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000

# Database
DATABASE_URL = "sqlite:///./voice_profiles.db"

# Model-specific settings
MODEL_CONFIGS = {
    "separation": {
        "convtasnet": {
            "use_gpu": True,
            "sample_rate": SAMPLE_RATE
        },
        "sepformer": {
            "use_gpu": True,
            "sample_rate": SAMPLE_RATE
        }
    },
    "embedding": {
        "resemblyzer": {
            "dimension": 256,
            "use_gpu": False
        },
        "speechbrain": {
            "dimension": 512,
            "use_gpu": False
        },
        "ecapa": {
            "dimension": 192,
            "use_gpu": False
        }
    },
    "vad": {
        "webrtcvad": {
            "aggressiveness": 3,
            "frame_duration": 30
        },
        "pyannote_vad": {
            "threshold": 0.5,
            "min_duration_on": 0.1,
            "min_duration_off": 0.1
        }
    }
}

# File size limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}

# Performance settings
CHUNK_SIZE = 1024
BATCH_SIZE = 32

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Ensure directories exist
def initialize_directories():
    """Create all required directories if they don't exist"""
    directories = [
        AUDIO_DIR, ISOLATED_DIR, MIXED_DIR, TEMP_DIR,
        EMBEDDINGS_DIR, PRETRAINED_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Environment variable overrides
def get_config(key: str, default=None):
    """Get configuration value with environment variable override"""
    env_key = f"LOUDCLEAR_{key.upper()}"
    return os.environ.get(env_key, default)