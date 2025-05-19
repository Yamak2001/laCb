# models/embedding/pyannote_model.py
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

class PyannoteSpeakerModel(BaseEmbeddingModel):
    """Pyannote.audio speaker embedding model"""
    def __init__(self, model_name: str, use_gpu: bool = False, 
                 pretrained_model: str = "pyannote/embedding"):
        super().__init__(model_name, use_gpu)
        self.device = None
        self.model = None
        self.pretrained_model = pretrained_model
        self.embedding_dim = 512  # Default for Pyannote embedding
    def load_model(self) -> None:
        """Load the Pyannote embedding model."""
        start_time = time.time()
        
        try:
            # Import here to avoid import errors for users without pyannote
            from pyannote.audio import Model
            from pyannote.audio.pipelines import SpeakerEmbedding
            
            logger.info(f"Loading Pyannote speaker embedding model from: {self.pretrained_model}")
            
            # Set device
            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
            device_name = "CUDA" if self.device.type == "cuda" else "CPU"
            logger.info(f"Using {device_name} for Pyannote model")
            
            # Load the pretrained model
            try:
                # First, try the newer API (pyannote.audio 2.x)
                embedding_model = Model.from_pretrained(
                    self.pretrained_model,
                    use_auth_token=True  # Requires Hugging Face token
                )
                
                # Create pipeline
                pipeline = SpeakerEmbedding(
                    embedding_model,
                    device=self.device
                )
                
                self.model = pipeline
                logger.info("Pyannote speaker embedding model loaded (new API)")
                
            except (ImportError, AttributeError, ValueError):
                logger.info("Falling back to older Pyannote API")
                
                try:
                    # Try older API (pyannote.audio 1.x)
                    from pyannote.audio.embedding.approaches import AngularAdditiveMarginLoss
                    from pyannote.audio.embedding.extractors import SincTDNN
                    
                    # Initialize the model
                    model = SincTDNN()
                    loss = AngularAdditiveMarginLoss()
                    
                    # Load pretrained weights
                    model_path = os.path.join(MODELS_DIR, "pyannote_embedding.torch")
                    if os.path.exists(model_path):
                        model.load_state_dict(torch.load(model_path, map_location=self.device))
                    else:
                        # Try to download from somewhere
                        logger.warning(f"Pretrained model not found at {model_path}")
                        # Fallback to basic initialization
                    
                    model = model.to(self.device)
                    model.eval()
                    
                    self.model = model
                    logger.info("Pyannote speaker embedding model loaded (old API)")
                    
                except (ImportError, Exception) as e:
                    logger.error(f"Failed to load Pyannote model with old API: {e}")
                    raise ValueError("Could not load Pyannote model with any available API")
            
                        
        except ImportError as e:
            logger.error(f"Failed to import Pyannote.audio: {e}")
            raise ImportError(f"Pyannote.audio is not installed. Install with pip install pyannote.audio: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load Pyannote embedding model: {e}")
            raise
    
    def create_embedding(self, audio_path: str) -> np.ndarray:
        """Create a speaker embedding from an audio file using Pyannote."""
        # Ensure the model is loaded
        if self.model is None:
            logger.info("Model not loaded yet, loading now")
            self.load_model()
        
        try:
            logger.info(f"Creating speaker embedding for: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Handle different Pyannote versions
            try:
                # Try newer version (pyannote.audio 2.x)
                if hasattr(self.model, 'embed_from_file'):
                    # Direct from file
                    embedding = self.model.embed_from_file(audio_path)
                elif hasattr(self.model, '__call__'):
                    # Pipeline call
                    embedding = self.model(audio_path)
                else:
                    # Fall back to older method
                    raise AttributeError("Model doesn't have expected methods")
                    
            except (AttributeError, Exception):
                # Try older version (pyannote.audio 1.x)
                logger.info("Using older Pyannote API for embedding")
                
                # Load the audio file
                from pyannote.audio.features import Precomputed
                from pyannote.core import Segment
                
                waveform, sample_rate = self._load_audio(audio_path)
                duration = waveform.shape[1] / sample_rate
                
                # Create embedding
                embedding = self.model(waveform.to(self.device)).cpu().numpy().squeeze()
                
            logger.info(f"Created embedding with shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating Pyannote speaker embedding: {e}")
            raise
    def _load_audio(self, audio_path: str) -> tuple:
        """
        Load audio file using the appropriate method for Pyannote.
        
        Returns:
            tuple: (waveform, sample_rate)
        """
        try:
            # Try torchaudio first (preferred by Pyannote)
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform, sample_rate
        except:
            # Fall back to librosa
            import librosa
            import torch
            
            waveform, sample_rate = librosa.load(audio_path, sr=None, mono=True)
            waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension
            return waveform, sample_rate
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two speaker embeddings."""
        try:
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            raise