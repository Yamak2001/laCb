#!/bin/bash
# install_all_dependencies.sh
# Script to install all dependencies for Loud & Clear, including all model types

set -e  # Exit on error

echo "===== Installing Dependencies for Loud & Clear ====="
echo "This script will install all required dependencies for the different models."
echo

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS_TYPE="Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="MacOS"
    elif [[ "$OSTYPE" == "cygwin" ]]; then
        OS_TYPE="Windows"
    elif [[ "$OSTYPE" == "msys" ]]; then
        OS_TYPE="Windows"
    elif [[ "$OSTYPE" == "win32" ]]; then
        OS_TYPE="Windows"
    else
        OS_TYPE="Unknown"
    fi
    echo "Detected OS: $OS_TYPE"
}

# Function to check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "Python version: $PYTHON_VERSION"
    
    # Check if pip is installed
    if ! command -v pip3 &> /dev/null; then
        echo "Error: pip is not installed. Please install pip."
        exit 1
    fi
}

# Function to check for virtual environment
check_venv() {
    # Check if we're in a virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "Warning: You are not in a virtual environment."
        read -p "Do you want to continue with installation in system Python? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting. Please activate a virtual environment and try again."
            exit 1
        fi
    else
        echo "Using virtual environment: $VIRTUAL_ENV"
    fi
}

# Function to install core dependencies
install_core() {
    echo "===== Installing Core Dependencies ====="
    
    # Fix for torch - use specific versions that work well together
    echo "Installing PyTorch and torchaudio..."
    if [[ "$OS_TYPE" == "MacOS" ]]; then
        # Check if it's Apple Silicon
        if [[ $(uname -m) == 'arm64' ]]; then
            echo "Detected Apple Silicon (M1/M2/M3) Mac"
            # Use specific versions known to work together for M1/M2/M3
            pip install torch==2.1.0 torchaudio==2.1.0
        else
            # For Intel Macs
            pip install torch==2.1.0 torchaudio==2.1.0
        fi
    else
        # For Linux/Windows
        pip install torch==2.1.0 torchaudio==2.1.0
    fi
    
    # Install other dependencies
    pip install -r requirements.txt
    
    echo "Core dependencies installed successfully."
    echo
}

# Function to install all SpeechBrain models
install_speechbrain() {
    echo "===== Installing SpeechBrain Dependencies ====="
    pip install speechbrain>=1.0.0
    pip install transformers>=4.11.3
    pip install hyperpyyaml>=0.0.1
    pip install huggingface-hub>=0.7.0
    echo "SpeechBrain dependencies installed successfully."
    echo
}

# Function to install Demucs and HDemucs dependencies
install_demucs() {
    echo "===== Installing Demucs Dependencies ====="
    pip install pyyaml
    pip install dora-search
    pip install demucs>=4.0.0
    pip install diffq>=0.2.0
    pip install julius>=0.2.3
    pip install torchmetrics>=0.8.0
    echo "Demucs dependencies installed successfully."
    echo
}

# Function to install Pyannote dependencies
install_pyannote() {
    echo "===== Installing Pyannote.audio Dependencies ====="
    pip install pyannote.audio>=2.1.1
    pip install pyannote.core>=4.1
    pip install pyannote.database>=4.1.1
    pip install pyannote.metrics>=3.2
    pip install einops>=0.6.0 
    pip install pytorch_lightning>=2.0.0
    
    echo "Pyannote.audio dependencies installed successfully."
    echo
    echo "IMPORTANT: To use Pyannote.audio, you need to:"
    echo "1. Create a HuggingFace account: https://huggingface.co/join"
    echo "2. Accept the user agreement: https://huggingface.co/pyannote/speaker-diarization"
    echo "3. Create an access token: https://huggingface.co/settings/tokens"
    echo ""
    echo "Then, you can authenticate with:"
    echo "python -c \"from huggingface_hub import login; login()\""
    echo
}

# Function to install Asteroid dependencies
install_asteroid() {
    echo "===== Installing Asteroid Dependencies ====="
    # Don't try to install asteroid-models which is causing issues
    pip install asteroid>=0.7.0
    pip install pb_bss_eval>=0.0.2
    pip install torch_stoi>=0.1.2
    echo "Asteroid dependencies installed successfully."
    echo
}

# Function to install TitaNet dependencies (fallback to resemblyzer)
install_titanet() {
    echo "===== Installing TitaNet Dependencies ====="
    echo "Note: Full NeMo installation is quite large, using minimal dependencies"
    pip install resemblyzer>=0.1.0
    echo "TitaNet fallback dependencies installed successfully."
    echo
}

# Function to install VAD dependencies
install_vad() {
    echo "===== Installing VAD Dependencies ====="
    pip install webrtcvad>=2.0.10
    echo "WebRTC VAD dependencies installed successfully."
    echo
}

# Function to install FFmpeg
install_ffmpeg() {
    echo "===== Installing FFmpeg ====="
    
    if [[ "$OS_TYPE" == "Linux" ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y ffmpeg
        elif command -v yum &> /dev/null; then
            sudo yum install -y ffmpeg
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y ffmpeg
        else
            echo "Please install FFmpeg manually for your Linux distribution."
        fi
    elif [[ "$OS_TYPE" == "MacOS" ]]; then
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "Please install Homebrew (https://brew.sh/) and then run 'brew install ffmpeg'"
        fi
    elif [[ "$OS_TYPE" == "Windows" ]]; then
        echo "Please download and install FFmpeg from https://ffmpeg.org/download.html"
        echo "Make sure to add FFmpeg to your PATH."
    else
        echo "Please install FFmpeg manually for your OS."
    fi
    
    # Check if installation was successful
    if command -v ffmpeg &> /dev/null; then
        echo "FFmpeg installed successfully."
    else
        echo "FFmpeg installation may not have been successful. Please install it manually."
    fi
    echo
}

# Function to create database
create_database() {
    echo "===== Setting up Database ====="
    echo "Creating database tables if they don't exist..."
    
    # Create tables using Python
    python -c 'from database import Base, engine; Base.metadata.create_all(engine)'
    
    echo "Database setup complete."
    echo
}

# Function to install the model implementation files
install_model_files() {
    echo "===== Installing Model Implementation Files ====="
    
    # Create directories if they don't exist
    mkdir -p models/separation
    mkdir -p models/embedding
    mkdir -p models/vad
    mkdir -p utils
    
    # Check if files exist in the current directory
    if [ -f "sepformer_model.py" ]; then
        echo "Copying model files from current directory..."
        # Copy separation models
        [ -f "sepformer_model.py" ] && cp sepformer_model.py models/separation/
        [ -f "hdemucs_model.py" ] && cp hdemucs_model.py models/separation/
        
        # Copy embedding models
        [ -f "improved-speechbrain-model.py" ] && cp improved-speechbrain-model.py models/embedding/speechbrain_model.py
        [ -f "improved-ecapa-model.py" ] && cp improved-ecapa-model.py models/embedding/ecapa_model.py
        [ -f "pyannote_model.py" ] && cp pyannote_model.py models/embedding/
        [ -f "titanet_model.py" ] && cp titanet_model.py models/embedding/
        
        # Copy error handling
        [ -f "error_handling.py" ] && cp error_handling.py utils/
        [ -f "improved-audio-processing.py" ] && cp improved-audio-processing.py utils/audio_processing.py
    else
        echo "Model implementation files not found in current directory."
        echo "Please make sure to copy them manually to the appropriate directories."
    fi
    
    echo "Model files installation complete."
    echo
}

# Main installation process
main() {
    detect_os
    check_python
    check_venv
    
    echo "This script will install the following components:"
    echo "1. Core dependencies (with specific PyTorch/torchaudio versions)"
    echo "2. FFmpeg (for audio processing)"
    echo "3. SpeechBrain (for voice embedding and VAD)"
    echo "4. Demucs (for voice separation)"
    echo "5. Pyannote.audio (for voice detection)"
    echo "6. Asteroid (for other separation models)"
    echo "7. WebRTC VAD (for voice activity detection)"
    echo
    echo "Additionally, it will set up the database tables."
    echo
    
    read -p "Do you want to continue with the installation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation aborted."
        exit 0
    fi
    
    # Proceed with installation
    install_core
    install_ffmpeg
    install_speechbrain
    install_demucs
    install_pyannote
    install_asteroid
    install_vad
    install_titanet
    install_model_files
    
    # Create database tables
    create_database
    
    echo "===== Installation Complete ====="
    echo "All dependencies have been installed."
    echo
    echo "You can now start the server with:"
    echo "uvicorn main:app --host 0.0.0.0 --port 8000"
    echo
}

# Run the installation
main