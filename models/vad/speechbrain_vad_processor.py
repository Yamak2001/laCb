# models/vad/speechbrain_vad_processor.py
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

class SpeechBrainVADProcessor(BaseVADProcessor):
    """SpeechBrain Voice Activity Detection processor."""
    
    def __init__(self, model_name: str, threshold: float = 0.5, 
                 pretrained_model: str = "speechbrain/vad-crdnn-libriparty"):
        super().__init__(model_name)
        self.threshold = threshold
        self.device = None
        self.model = None
        self.pretrained_model = pretrained_model
    
    def load_model(self) -> None:
        """Load the SpeechBrain VAD model."""
        try:
            # Import here to avoid import errors for users without speechbrain
            try:
                # For SpeechBrain 1.0 and newer
                from speechbrain.inference import VAD
            except ImportError:
                # For older SpeechBrain versions
                from speechbrain.pretrained import VAD
            
            logger.info(f"Loading SpeechBrain VAD model from: {self.pretrained_model}")
            
            # Set device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_name = "CUDA" if self.device.type == "cuda" else "CPU"
            logger.info(f"Using {device_name} for SpeechBrain VAD model")
            
            # Load the pretrained model
            self.model = VAD.from_hparams(
                source=self.pretrained_model,
                savedir=os.path.join(MODELS_DIR, "speechbrain_vad"),
                run_opts={"device": self.device}
            )
            
            logger.info("SpeechBrain VAD model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import SpeechBrain: {e}")
            raise ImportError(f"SpeechBrain is not installed. Install with pip install speechbrain: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain VAD model: {e}")
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
            output_path = os.path.join(base_dir, f"{base_name}_speechbrain_vad.wav")
        
        try:
            # Load the audio file
            logger.info(f"Loading audio file: {audio_path}")
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)  # SpeechBrain VAD expects 16kHz
            logger.info(f"Loaded audio with duration {len(audio)/sr:.2f}s")
            
            # Run VAD inference
            logger.info("Running SpeechBrain VAD")
            
            # We need to save the audio in WAV format for SpeechBrain to read it
            temp_wav_path = os.path.join(os.path.dirname(output_path), f"temp_vad_{os.path.basename(audio_path)}")
            sf.write(temp_wav_path, audio, sr)
            
            try:
                # Check if the model supports the threshold parameter
                # Try with newer API signature first
                speech_segments = self.model.get_speech_segments(
                    temp_wav_path,
                    activation_threshold=self.threshold,  # For newer versions
                    min_speech_duration_ms=250,
                    max_speech_duration_s=float("inf"),
                    min_silence_duration_ms=100
                )
            except TypeError:
                try:
                    # Try older API without the threshold parameter
                    speech_segments = self.model.get_speech_segments(
                        temp_wav_path,
                        min_speech_duration_ms=250,
                        max_speech_duration_s=float("inf"),
                        min_silence_duration_ms=100
                    )
                except Exception as e:
                    # Fall back to even older API for maximum compatibility
                    speech_segments = self.model.get_speech_segments(temp_wav_path)
            
            # Clean up temporary file
            try:
                os.remove(temp_wav_path)
            except:
                pass
            
            # Create a mask for speech segments
            mask = np.zeros_like(audio, dtype=bool)
            
            # Process each speech segment
            for segment in speech_segments:
                start_sample = int(segment[0] * sr)
                end_sample = min(int(segment[1] * sr), len(audio))
                
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
            logger.error(f"Error in SpeechBrain VAD processing: {e}")
            self.metrics["processing_time"] = time.time() - start_time
            raise