# models/separation/sepformer_model.py
import os
import numpy as np
import torch
import librosa
import soundfile as sf
import time
import logging
import config
from typing import List, Dict, Any

from models.base import BaseSeparationModel

# Set up logging
logger = logging.getLogger(__name__)

# Directory setup
MODELS_DIR = config.PRETRAINED_DIR
TEMP_DIR = "data/temp"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

class SepFormerModel(BaseSeparationModel):
    """SepFormer model for audio source separation"""
    
    def __init__(self, model_name: str, use_gpu: bool = False, sample_rate: int = 16000):
        super().__init__(model_name, use_gpu)
        self.sample_rate = sample_rate
        self.device = None
        self.model = None
    
    def load_model(self) -> None:
        """Load a pre-trained SepFormer model for voice separation."""
        start_time = time.time()
        
        try:
            # Try to import the Asteroid models
            try:
                from asteroid.models import SepFormer
                self._load_asteroid_model()
            except ImportError:
                logger.warning("Asteroid SepFormer not found. Trying alternative approach")
                self._load_via_torch_hub()
            
            self.metrics["load_time"] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"Failed to load SepFormer model: {e}")
            self._create_fallback_model()
            self.metrics["load_time"] = time.time() - start_time
    
    def _load_asteroid_model(self):
        """Load model using Asteroid library"""
        from asteroid.models import SepFormer
        
        max_retries = 3
        model_sources = [
            "asteroid/sepformer_wham_enhancesingle_16k",
            "speechbrain/sepformer-whamr-enhancement",
            "JorisCos/sepformer_libri2mix_sepclean_16k",
        ]
        
        for source in model_sources:
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to load SepFormer model from {source} (attempt {attempt+1}/{max_retries})")
                    model = SepFormer.from_pretrained(
                        source,
                        cache_dir=MODELS_DIR
                    )
                    logger.info("SepFormer model loaded successfully!")
                    
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
        
        # If all sources fail, try the torch hub approach
        logger.warning("All Asteroid model loading attempts failed. Trying torch.hub")
        self._load_via_torch_hub()
    
    def _load_via_torch_hub(self):
        """Load model using torch.hub (alternative method)"""
        try:
            logger.info("Attempting to load SepFormer via torch.hub")
            
            # Try loading from torch hub
            model = torch.hub.load(
                'asteroid-team/asteroid',
                'sepformer_wham_enhancesingle_16k',
                cache_dir=MODELS_DIR
            )
            
            logger.info("SepFormer model loaded via torch.hub!")
            
            # Set device
            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
            model.to(self.device)
            model.eval()
            
            self.model = model
            return
            
        except Exception as e:
            logger.error(f"Failed to load model via torch.hub: {e}")
            self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a fallback SepFormer model"""
        try:
            logger.warning("Creating fallback SepFormer model. Performance may be reduced.")
            
            # Try to import the necessary class
            try:
                from asteroid.models import SepFormer
            except ImportError:
                # If Asteroid is not available, try to install it
                logger.warning("Asteroid not found. Attempting to install...")
                import subprocess
                subprocess.run(["pip", "install", "asteroid-models"], check=True)
                from asteroid.models import SepFormer
            
            # Create a smaller model for faster loading
            model = SepFormer(
                n_src=2,  # Number of sources to separate
                enc_dim=64,
                feature_dim=64,
                hidden_dim=128,
                layer=2,
                nhead=4,
                ff_dim=256,
                chunk_size=100,
                hop_size=50,
                norm_type="gLN"
            )
            
            self.device = torch.device("cpu")  # Use CPU for fallback
            model.to(self.device)
            model.eval()
            
            logger.info("Created fallback SepFormer model")
            self.model = model
            
        except Exception as e:
            logger.error(f"Failed to create fallback SepFormer model: {e}")
            logger.critical("SepFormer model is not available. Voice separation may fail.")
            self.model = None
    
    @BaseSeparationModel.track_performance
    def separate_sources(self, audio_path: str, output_dir: str) -> List[str]:
        """Separate audio sources from a mixed input file using the SepFormer model."""
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
            
            # If still None after loading attempt, fail gracefully
            if self.model is None:
                raise RuntimeError("SepFormer model could not be loaded. Try a different separation model.")
        
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
            logger.info("Processing audio with SepFormer model")
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Separate sources
            with torch.no_grad():
                model_output = self.model(audio_tensor)
            
            # Handle different output formats
            if isinstance(model_output, tuple):
                # Some models return (output, metrics)
                separated = model_output[0].cpu().numpy()
            else:
                # Others just return the output
                separated = model_output.cpu().numpy()
            
            # Handle dimension differences
            if separated.ndim == 4:  # [batch, channel, source, time]
                separated = separated.squeeze(0).squeeze(0)  # Remove batch and channel dims
            elif separated.ndim == 3:  # [batch, source, time]
                separated = separated.squeeze(0)  # Remove batch dim
            
            # Log separated output properties
            logger.info(f"Separated output shape: {separated.shape}")
            
            # Save separated sources
            output_paths = []
            for idx in range(separated.shape[0]):
                source = separated[idx]
                output_file = os.path.join(output_dir, f"{base_filename}_source_{idx+1}.wav")
                
                # Normalize output
                source = source / np.max(np.abs(source)) * 0.9
                
                # Save to file
                sf.write(output_file, source, self.sample_rate)
                output_paths.append(output_file)
                logger.info(f"Saved source {idx+1} to {output_file}")
            
            return output_paths
            
        except Exception as e:
            logger.error(f"Error in SepFormer audio source separation: {e}")
            raise