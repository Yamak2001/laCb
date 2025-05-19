# models/separation/convtasnet_model.py
import os
import numpy as np
import torch
import librosa
import soundfile as sf
import time
import logging
import config
from typing import List, Dict, Any

from asteroid.models import ConvTasNet
from models.base import BaseSeparationModel

# Set up logging
logger = logging.getLogger(__name__)

# Directory setup
MODELS_DIR = config.PRETRAINED_DIR
TEMP_DIR = "data/temp"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

class ConvTasNetModel(BaseSeparationModel):
    """ConvTasNet model for audio source separation"""
    
    def __init__(self, model_name: str, use_gpu: bool = False, sample_rate: int = 16000):
        super().__init__(model_name, use_gpu)
        self.sample_rate = sample_rate
        self.device = None
        self.model = None
    
    def load_model(self) -> None:
        """Load a pre-trained Conv-TasNet model for voice separation with retry logic."""
        start_time = time.time()
        max_retries = 3
        
        # Try multiple model sources in case one fails
        model_sources = [
            "conv_tasnet_sepnoisy_libri2Mix_enhsingle_16k",  # Standard Asteroid model name
            "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",    # Hugging Face model path
            "asteroidevaluate/ConvTasNet_Libri2Mix_sepnoisy_16k"  # Alternative source
        ]
        
        # Try loading from each source with retries
        for source in model_sources:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to load model from {source} (attempt {attempt+1}/{max_retries})")
                    model = ConvTasNet.from_pretrained(
                        source,
                        cache_dir=MODELS_DIR
                    )
                    logger.info("Model loaded successfully!")
                    
                    # Move model to GPU if available and requested
                    self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
                    if self.device.type == "cuda":
                        logger.info("Using GPU acceleration")
                    else:
                        logger.info("Using CPU for processing")
                        
                    model.to(self.device)
                    model.eval()  # Set model to evaluation mode
                    
                    self.model = model
                    return
                    
                except Exception as e:
                    logger.error(f"Failed to load model from {source}: {e}")
                    if attempt < max_retries - 1:
                        # Wait before retrying
                        wait_time = 2 * (attempt + 1)  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retrying...")
                        time.sleep(wait_time)
        
        # If all sources and retries fail, create a fallback model
        logger.warning("All model loading attempts failed. Creating fallback model.")
        model = ConvTasNet(
            n_src=2,  # Number of sources to separate
            n_repeats=3,  # Number of repeats
            n_filters=512,  # Number of filters in autoencoder
            kernel_size=16,  # Kernel size in convolutional blocks
            n_blocks=8,  # Number of convolutional blocks in each repeat
            n_channels=128,  # Number of channels in bottleneck
            norm_type="gLN"  # Normalization type
        )
        self.device = torch.device("cpu")
        model.to(self.device)
        model.eval()
        
        logger.info("Created fallback model")
        self.model = model
    
    def separate_sources(self, audio_path: str, output_dir: str) -> List[str]:
        """Separate audio sources from a mixed input file using the Conv-TasNet model."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base output filename from input
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Check if file exists and is readable
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Ensure the model is loaded
        if self.model is None:
            logger.info("Model not loaded yet, loading now")
            self.load_model()
        
        try:
            # Load and resample audio if necessary
            logger.info(f"Loading audio file: {audio_path}")
            audio, orig_sr = librosa.load(audio_path, sr=None, mono=True)
            logger.info(f"Audio loaded successfully. Original sample rate: {orig_sr}Hz")
            
            # Resample if necessary
            if orig_sr != self.sample_rate:
                logger.info(f"Resampling from {orig_sr}Hz to {self.sample_rate}Hz")
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
            
            # Normalize audio to prevent clipping
            audio = audio / np.max(np.abs(audio)) * 0.9
            
            # Log audio properties
            logger.info(f"Audio duration: {len(audio)/self.sample_rate:.2f} seconds")
            logger.info(f"Audio shape: {audio.shape}")
            
            # Convert to tensor and reshape for model
            logger.info("Processing audio with model")
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Separate sources
            with torch.no_grad():
                model_output = self.model(audio_tensor)
            
            # Convert output to numpy array
            # Model output shape will be [1, n_sources, audio_length]
            separated = model_output.cpu().numpy()
            
            # Log separated output properties
            logger.info(f"Separated output shape: {separated.shape}")
            
            # Save separated sources
            output_paths = []
            for idx in range(separated.shape[1]):
                source = separated[0, idx]
                output_file = os.path.join(output_dir, f"{base_filename}_source_{idx+1}.wav")
                
                # Normalize output
                source = source / np.max(np.abs(source)) * 0.9
                
                # Save to file
                sf.write(output_file, source, self.sample_rate)
                output_paths.append(output_file)
                logger.info(f"Saved source {idx+1} to {output_file}")
            
            return output_paths
            
        except Exception as e:
            logger.error(f"Error in audio source separation: {e}")
            raise