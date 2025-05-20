# Loud & Clear: Voice Isolation System

Loud & Clear is a modular voice isolation system that extracts specific voices from mixed audio using deep learning models. The system supports multiple separation and embedding models, allowing you to choose the best combination for your needs.

## Features

- **Multiple Voice Separation Models**: ConvTasNet, DPRNN, SepFormer
- **Voice Embedding Models**: Resemblyzer, SpeechBrain, ECAPA-TDNN, Pyannote, TitaNet
- **Voice Activity Detection**: WebRTC VAD, SpeechBrain VAD, Pyannote VAD
- **REST API**: Easy integration with your applications
- **Fallback Mechanisms**: Robust handling of model failures
- **Voice Profile Management**: Store and retrieve voice embeddings

## System Requirements

- Python 3.8+ 
- FFmpeg (for audio conversion)
- 8GB+ RAM (16GB+ recommended for larger models)
- CUDA-compatible GPU or Apple Silicon Mac (optional, for faster processing)
- 5GB+ disk space for models and dependencies

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/loud-and-clear.git
cd loud-and-clear
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e .  # Install in development mode
# OR
pip install .     # Install normally
```

For GPU support:
```bash
# For NVIDIA GPUs (CUDA)
pip install -e ".[gpu]"

# For Apple Silicon Macs (MPS)
# PyTorch 2.0+ with MPS support is included in the default installation
```

For development:
```bash
pip install -e ".[dev]"
```

### 4. Initialize Directories

```bash
python init_directories.py
```

### 5. Set Up the Database

```bash
python -c "from database import Base, engine; Base.metadata.create_all(engine)"
```

## Configuration

The system uses a centralized configuration in `config.py`. You can override settings using environment variables:

```bash
export LOUDCLEAR_API_KEY="your-secret-key"
export LOUDCLEAR_CORS_ORIGIN="http://localhost:3000"
```

## Running the Server

Start the development server:

```bash
uvicorn main:app --reload
```

Start the production server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000

## API Usage

### Create a Voice Profile

```bash
curl -X POST "http://localhost:8000/api/voice-profiles/" \
  -H "Content-Type: multipart/form-data" \
  -F "name=JohnDoe" \
  -F "category=personal" \
  -F "audio_file=@/path/to/clean_voice_sample.wav"
```

### Process Mixed Audio

```bash
curl -X POST "http://localhost:8000/api/audio/process" \
  -H "Content-Type: multipart/form-data" \
  -F "profile_id=1" \
  -F "audio_file=@/path/to/mixed_audio.wav" \
  -F "separation_model=convtasnet" \
  -F "embedding_model=speechbrain" \
  -F "use_vad=true" \
  -F "vad_processor=webrtcvad"
```


## Model Recommendations

### Separation Models

- **ConvTasNet**: Fast, general-purpose separation
- **DPRNN**: Good for speech-only content
- **SepFormer**: Best quality but requires more memory

### Embedding Models

- **Resemblyzer**: Fast and lightweight
- **SpeechBrain**: Higher quality speaker identification
- **ECAPA-TDNN**: Best quality but slower
- **Pyannote**: Good for speaker diarization
- **TitaNet**: High performance on difficult cases

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
flake8 .
```

### Type Checking

```bash
mypy .
```

## Troubleshooting

### GPU Issues

Check CUDA availability:
```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Force CPU mode:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Model Loading Errors

Clear model cache:
```bash
rm -rf models/pretrained/*
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Acknowledgments

This project uses several open-source deep learning models:
- [Asteroid](https://github.com/asteroid-team/asteroid)
- [SpeechBrain](https://github.com/speechbrain/speechbrain)
- [TorchAudio](https://github.com/pytorch/audio)
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)