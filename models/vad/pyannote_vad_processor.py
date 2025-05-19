# models/vad/pyannote_vad_processor.py
import os
import numpy as np
import torch
import librosa
import soundfile as sf
import time
import logging
from typing import List, Dict, Any

from models.base import BaseVADProcessor

# Set up logging
logger = logging.getLogger(__name__)

# Directory setup
MODELS_DIR = "models/pretrained"
os.makedirs(MODELS_DIR, exist_ok=True)

class PyannoteSpeechDetectionProcessor(BaseVADProcessor):
    """Pyannote.audio Voice Activity Detection processor."""
    
    def __init__(self, model_name: str, threshold: float = 0.5, min_duration_on: float = 0.1, 
                 min_duration_off: float = 0.1):
        super().__init__(model_name)
        self.threshold = threshold
        self.min_duration_on = min_duration_on  # Minimum speech duration in seconds
        self.min_duration_off = min_duration_off  # Minimum silence duration in seconds
        self.device = None
        self.model = None
    
    def load_model(self) -> None:
        """Load the Pyannote VAD model."""
        try:
            # Import here to avoid import errors for users without pyannote
            from pyannote.audio import Pipeline
            
            logger.info("Initializing Pyannote VAD model")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load the pipeline
            try:
                # Try to load from Hugging Face
                pipeline = Pipeline.from_pretrained(
                    "pyannote/voice-activity-detection",
                    use_auth_token=True  # May require HuggingFace token
                )
            except Exception as e:
                logger.warning(f"Failed to load from Hugging Face: {e}")
                # Try to load from local cache
                pipeline = Pipeline.from_pretrained(
                    os.path.join(MODELS_DIR, "pyannote_vad")
                )
            
            pipeline = pipeline.to(self.device)
            
            # Set VAD parameters
            pipeline.instantiate({
                "onset": self.threshold,
                "offset": self.threshold,
                "min_duration_on": self.min_duration_on,
                "min_duration_off": self.min_duration_off
            })
            
            self.model = pipeline
            logger.info("Pyannote VAD model initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import pyannote.audio: {e}")
            raise ImportError(f"pyannote.audio is not installed. Install with pip install pyannote.audio: {e}")
        
        except Exception as e:
            logger.error(f"Failed to initialize Pyannote VAD model: {e}")
            raise
    
    def process_audio(self, audio_path: str, output_path: str = None) -> str:
        """
        Process audio to detect voice activity and filter non-speech segments.
        
        Args:
            audio_path (str): Path to input audio file
            output_path (str): Path to save processed audio
            
        Returns:
            str: Path to processed audio file
        """
        start_time = time.time()
        
        # Ensure the model is loaded
        if self.model is None:
            self.load_model()
        
        # Create output path if not provided
        if output_path is None:
            base_dir = os.path.dirname(audio_path)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_pyannote_vad.wav")
        
        try:
            # Run VAD inference
            logger.info(f"Running Pyannote VAD on: {audio_path}")
            vad_result = self.model(audio_path)
            
            # Load the audio file
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            logger.info(f"Loaded audio with sample rate {sr}Hz, duration {len(audio)/sr:.2f}s")
            
            # Create mask for speech segments
            mask = np.zeros_like(audio, dtype=bool)
            
            # Go through each speech region and mark it in the mask
            for segment, _, _ in vad_result.itertracks(yield_label=True):
                start_sample = int(segment.start * sr)
                end_sample = min(int(segment.end * sr), len(audio))
                
                if start_sample < end_sample:
                    mask[start_sample:end_sample] = True
            
            # Extract only the speech segments
            speech_audio = audio[mask]
            
            # If no speech detected, return minimal audio to avoid errors
            if len(speech_audio) == 0:
                logger.warning("No speech detected in the audio file")
                # Return a small segment of the original audio to avoid errors
                speech_audio = audio[:min(len(audio), int(0.5 * sr))]
            
            # Save processed audio
            logger.info(f"Saving processed audio to: {output_path}")
            sf.write(output_path, speech_audio, sr)
            
            # Update metrics
            self.metrics["processing_time"] = time.time() - start_time
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in Pyannote VAD processing: {e}")
            self.metrics["processing_time"] = time.time() - start_time
            raise