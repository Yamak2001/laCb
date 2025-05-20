# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
Loud & Clear is a modular voice isolation system that extracts specific voices from mixed audio using deep learning models. The system is built with FastAPI and provides REST API endpoints for voice profile management and audio processing.

## Architecture

### Core Components
- **Models**: Modular architecture with base classes for separation, embedding, and VAD processing
- **Routers**: FastAPI endpoints for voice profiles, audio processing, and monitoring
- **Database**: SQLAlchemy for voice profile persistence

### Model System
- `models/base.py`: Abstract base classes (BaseSeparationModel, BaseEmbeddingModel, BaseVADProcessor)
- `models/factory.py`: Factory pattern for dynamic model registration and instantiation  
- Separation models: ConvTasNet, DPRNN, SepFormer
- Embedding models: Resemblyzer, SpeechBrain, ECAPA-TDNN, Pyannote, TitaNet
- VAD processors: WebRTC VAD, SpeechBrain VAD, Pyannote VAD

## Common Commands

### Development Server
```bash
# Start the development server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run in production
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Dependencies
```bash
# Install all dependencies (recommended)
chmod +x install_dependencies.sh
./install_dependencies.sh

# Install specific model dependencies  
./install_speechbrain_dependencies.sh
# Demucs dependencies removed
./install_pyannote_dependencies.sh

# Manual installation
pip install -r requirements.txt
```

### Database Setup
```bash
# Initialize database tables
python -c 'from database import Base, engine; Base.metadata.create_all(engine)'

# Hugging Face authentication (for Pyannote models)
python -c "from huggingface_hub import login; login()"
```

### Directory Initialization
```bash
# Create required directory structure
python init_directories.py
```

## Key File Patterns

### Adding New Models
1. Create model class inheriting from appropriate base class in `models/base.py`
2. Implement required abstract methods
3. Register model in `models/factory.py` using the factory pattern
4. Add model-specific dependencies to requirements.txt

### API Endpoints
- Voice profiles: `/api/voice-profiles/` (CRUD operations)
- Audio processing: `/api/audio/process` (voice isolation)

### Configuration
- API key for monitoring: `API_KEY` in `routers/monitoring.py`
- Model registration: `register_all_models()` in `models/factory.py`
- Directories: Configured in `init_directories.py`

## Important Considerations
- Use GPU support by setting `use_gpu=true` in API calls
  - CUDA is automatically used on NVIDIA GPUs
  - MPS (Metal Performance Shaders) is automatically used on Apple Silicon Macs
- Large models require 8GB+ RAM (16GB+ recommended)
- Pyannote models require Hugging Face authentication
- FFmpeg is required for audio conversion
- Temporary files are stored in `data/temp/`