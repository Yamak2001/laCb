# models/embedding/ecapa_model.py
import os
import numpy as np
import torch
import time
import logging
from typing import List, Dict, Any

from models.base import BaseEmbeddingModel

# Set up logging
logger = logging.getLogger(__name__)

# Directory setup
MODELS_DIR = "models/pretrained"
os.makedirs(MODELS_DIR, exist_ok=True)

class ECAPAModel(BaseEmbeddingModel):
    """ECAPA-TDNN model for voice embedding creation and comparison (SpeechBrain)"""
    def __init__(self, model_name: str, use_gpu: bool = False, 
                 pretrained_model: str = "speechbrain/spkrec-ecapa-voxceleb"):
        super().__init__(model_name, use_gpu)
        self.device = None
        self.model = None
        self.pretrained_model = pretrained_model
        self.embedding_dim = 192  # Default for ECAPA-TDNN
    def load_model(self) -> None:
        """Load the ECAPA-TDNN model from SpeechBrain."""
        start_time = time.time()
        
        try:
            # Import here to avoid import errors for users without speechbrain
            from speechbrain.pretrained import EncoderClassifier
            
            logger.info(f"Loading ECAPA-TDNN model from: {self.pretrained_model}")
            
            # Set device
            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
            device_name = "CUDA" if self.device.type == "cuda" else "CPU"
            logger.info(f"Using {device_name} for ECAPA-TDNN model")
            
            # Load the pretrained model
            self.model = EncoderClassifier.from_hparams(
                source=self.pretrained_model,
                savedir=os.path.join(MODELS_DIR, "ecapa_tdnn"),
                run_opts={"device": self.device}
            )
            
            logger.info("ECAPA-TDNN model loaded successfully")
                        
        except ImportError as e:
            logger.error(f"Failed to import SpeechBrain: {e}")
            raise ImportError(f"SpeechBrain is not installed. Install with pip install speechbrain: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load ECAPA-TDNN model: {e}")
            raise
        def create_embedding(self, audio_path: str) -> np.ndarray:
        """Create a voice embedding from an audio file using ECAPA-TDNN."""
        # Ensure the model is loaded
        if self.model is None:
            logger.info("Model not loaded yet, loading now")
            self.load_model()
        
        try:
            logger.info(f"Creating voice embedding for: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load the audio file using torchaudio
            import torchaudio
            signal, fs = torchaudio.load(audio_path)
            
            # Convert to mono if needed
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim=0, keepdim=True)
                
            # Move to device
            signal = signal.to(self.device)
            
            # Extract the embedding using the loaded signal
            embeddings = self.model.encode_batch(signal)
            
            # Convert from tensor to numpy array
            embedding = embeddings.squeeze().cpu().numpy()
            
            logger.info(f"Created embedding with shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating ECAPA-TDNN voice embedding: {e}")
            # Try alternative method if the first one fails
            try:
                logger.info("Trying alternative embedding method")
                wav_info = torchaudio.info(audio_path)
                
                # Extract the embedding using the file path directly
                # Some versions of SpeechBrain support this method
                embeddings = self.model.encode_file(audio_path)
                
                # Convert from tensor to numpy array
                embedding = embeddings.squeeze().cpu().numpy()
                
                logger.info(f"Created embedding with shape: {embedding.shape}")
                return embedding
            except Exception as e2:
                logger.error(f"Alternative method also failed: {e2}")
                # Fall back to resemblyzer if available
                try:
                    logger.info("Trying Resemblyzer as fallback")
                    from resemblyzer import VoiceEncoder, preprocess_wav
                    
                    wav = preprocess_wav(audio_path)
                    fallback_encoder = VoiceEncoder()
                    embedding = fallback_encoder.embed_utterance(wav)
                    
                    logger.info(f"Created fallback embedding with shape: {embedding.shape}")
                    return embedding
                except Exception as e3:
                    logger.error(f"All embedding methods failed: {e3}")
                    raise RuntimeError("Could not create embeddings with any available method")
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two voice embeddings."""
        try:
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise