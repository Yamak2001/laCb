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
- CUDA-compatible GPU (optional, for faster processing)
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

The easiest way to install all dependencies is using the provided installation script:

```bash
chmod +x install_all_dependencies.sh
./install_all_dependencies.sh
```

This script will install all required dependencies, including:
- Core requirements
- PyTorch (with GPU support if available)
- FFmpeg
- Model-specific dependencies

If you prefer to install dependencies manually:

```bash
pip install -r requirements.txt

# For specific models:
pip install speechbrain>=1.0.0  # For SpeechBrain models
# Demucs model removed from the project
pip install asteroid>=0.7.0     # For ConvTasNet and DPRNN models
pip install pyannote.audio>=2.1.1  # For Pyannote models
pip install webrtcvad>=2.0.10   # For WebRTC VAD
```

### 4. Set Up the Database

```bash
python -c 'from database import Base, engine; Base.metadata.create_all(engine)'
```

### 5. Hugging Face Authentication (for Pyannote models)

```bash
python -c "from huggingface_hub import login; login()"
```

You'll need to:
1. Create a Hugging Face account at https://huggingface.co/join
2. Accept the user agreements for needed models:
   - https://huggingface.co/pyannote/speaker-diarization
   - https://huggingface.co/pyannote/segmentation
3. Create an access token at https://huggingface.co/settings/tokens

## Running the Server

Start the server with:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000

## Using the API

### 1. Create a Voice Profile

First, create a voice profile from a clean audio sample of the target voice:

```bash
curl -X POST "http://localhost:8000/api/voice-profiles/" \
  -H "Content-Type: multipart/form-data" \
  -F "name=JohnDoe" \
  -F "category=personal" \
  -F "audio_file=@/path/to/clean_voice_sample.wav"
```

### 2. Process Mixed Audio

Process mixed audio to isolate the voice:

```bash
curl -X POST "http://localhost:8000/api/audio/process" \
  -H "Content-Type: multipart/form-data" \
  -F "profile_id=1" \
  -F "audio_file=@/path/to/mixed_audio.wav" \
  -F "separation_model=convtasnet" \
  -F "embedding_model=speechbrain" \
  -F "use_vad=true" \
  -F "vad_processor=webrtcvad" \
  -F "use_gpu=false" \
  -F "save_metrics=true"
```


## Model Recommendations

### Separation Models

- **ConvTasNet**: Fast, general-purpose separation, good balance of quality and speed
- **DPRNN**: Good for speech-only content
- **SepFormer**: Best quality but requires more memory and computation

### Embedding Models

- **Resemblyzer**: Fast and lightweight, good for most cases
- **SpeechBrain**: Higher quality speaker identification
- **ECAPA-TDNN**: Best quality but slower
- **Pyannote**: Good for speaker diarization scenarios
- **TitaNet**: High performance on difficult cases

### VAD Processors

- **WebRTC VAD**: Fast and lightweight
- **SpeechBrain VAD**: Better accuracy
- **Pyannote VAD**: Best accuracy but slowest

## Troubleshooting

### Missing Dependencies

If you encounter missing dependency errors, install them individually:

```bash
pip install <missing_package>
```

### GPU Issues

If you're having trouble with GPU acceleration:

```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Force CPU mode if needed
python -c "import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''"
```

### Model Loading Errors

If models fail to load, check the cache directories:

```bash
ls -la models/pretrained
```

You may need to manually download model files or clear the cache.

## License

[MIT License](LICENSE)

## Acknowledgments

This project uses several open-source deep learning models:
- [Asteroid](https://github.com/asteroid-team/asteroid) - Audio Source Separation toolkit
- [SpeechBrain](https://github.com/speechbrain/speechbrain) - Speech processing toolkit
- [TorchAudio](https://github.com/pytorch/audio) - Audio processing toolkit
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) - Voice embedding
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) - Voice activity detection