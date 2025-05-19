# models/embedding/titanet_model.py
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

class TitaNetModel(BaseEmbeddingModel):
    """TitaNet model for speaker embedding (NVIDIA NeMo)"""
    def __init__(self, model_name: str, use_gpu: bool = False, 
                 pretrained_model: str = "titanet_large"):
        super().__init__(model_name, use_gpu)
        self.device = None
        self.model = None
        self.pretrained_model = pretrained_model
        self.embedding_dim = 192  # Default for TitaNet
    def load_model(self) -> None:
        """Load the TitaNet model from NVIDIA NeMo."""
        start_time = time.time()
        
        try:
            # Try to import NeMo
            try:
                import nemo
                import nemo.collections.asr as nemo_asr
                self._load_nemo_model()
            except ImportError:
                logger.warning("NeMo toolkit not found. Trying torch hub approach")
                self._load_via_torch_hub()
            
                        
        except Exception as e:
            logger.error(f"Failed to load TitaNet model: {e}")
            raise
    def _load_nemo_model(self):
        """Load model using NVIDIA NeMo toolkit"""
        import nemo.collections.asr as nemo_asr
        
        logger.info(f"Loading TitaNet model from NeMo: {self.pretrained_model}")
        
        try:
            # Set device
            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
            
            # Load the pretrained model
            model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name=self.pretrained_model,
                map_location=self.device
            )
            
            model.eval()
            self.model = model
            logger.info(f"TitaNet model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load TitaNet from NeMo: {e}")
            raise
    def _load_via_torch_hub(self):
        """Load TitaNet model using torch hub (alternative)"""
        try:
            logger.info("Attempting to load TitaNet via torch.hub")
            
            # Set device
            self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
            
            # Try to load model from Torch Hub
            model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'titanet', 
                                   map_location=self.device)
            
            model.eval()
            self.model = model
            logger.info(f"TitaNet model loaded via torch.hub on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load TitaNet via torch.hub: {e}")
            
            # Try to create a custom fallback
            self._create_fallback_model()
    def _create_fallback_model(self):
        """Create a simplified fallback model for basic functionality"""
        try:
            logger.warning("Creating simplified fallback model. Performance will be limited.")
            
            # This is a very basic implementation that won't perform well
            # In a real scenario, you would download weights or use a different model
            
            # Try to use a pretrained Resemblyzer model as fallback
            from resemblyzer import VoiceEncoder
            
            fallback = VoiceEncoder()
            self.model = fallback
            logger.info("Using Resemblyzer as fallback for TitaNet")
            
            # Flag that we are using the fallback
            self.using_fallback = True
            
        except Exception as e:
            logger.error(f"Failed to create fallback model: {e}")
            raise ValueError("Could not load or create a TitaNet model. Try using a different embedding model.")
        def create_embedding(self, audio_path: str) -> np.ndarray:
        """Create a speaker embedding from an audio file using TitaNet."""
        # Ensure the model is loaded
        if self.model is None:
            logger.info("Model not loaded yet, loading now")
            self.load_model()
        
        try:
            logger.info(f"Creating speaker embedding for: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Check if we're using the fallback model
            if hasattr(self, 'using_fallback') and self.using_fallback:
                logger.info("Using fallback model for embedding")
                return self._create_embedding_with_fallback(audio_path)
            
            # TitaNet in NeMo has a specific method for embedding
            if hasattr(self.model, 'get_embedding'):
                # NeMo TitaNet specific methods
                embedding = self.model.get_embedding(audio_path)
                
                # Convert to numpy
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                
            elif hasattr(self.model, 'embed_file'):
                # Torch Hub version might have different interface
                embedding = self.model.embed_file(audio_path)
                
                # Convert to numpy
                if isinstance(embedding, torch.Tensor):
                    embedding = embedding.cpu().numpy()
                    
            else:
                # Generic approach - load the audio and process it
                waveform, sample_rate = self._load_audio(audio_path)
                waveform = waveform.to(self.device)
                
                # Generic forward pass
                with torch.no_grad():
                    embedding = self.model(waveform)
                
                # Convert to numpy
                embedding = embedding.cpu().numpy().squeeze()
            
            logger.info(f"Created embedding with shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating TitaNet embedding: {e}")
            
            # Try fallback if main method failed
            if not hasattr(self, 'using_fallback'):
                logger.info("Trying fallback approach")
                try:
                    self.using_fallback = True
                    return self._create_embedding_with_fallback(audio_path)
                except Exception as fallback_error:
                    logger.error(f"Fallback approach failed: {fallback_error}")
                    
            raise
    def _create_embedding_with_fallback(self, audio_path: str) -> np.ndarray:
        """Create embedding using fallback method"""
        from resemblyzer import VoiceEncoder, preprocess_wav
        
        # Initialize encoder if needed
        if not isinstance(self.model, VoiceEncoder):
            self.model = VoiceEncoder()
        
        # Process with Resemblyzer
        wav = preprocess_wav(audio_path)
        embedding = self.model.embed_utterance(wav)
        
        logger.info(f"Created fallback embedding with shape: {embedding.shape}")
        return embedding
    def _load_audio(self, audio_path: str) -> tuple:
        """
        Load audio file for TitaNet processing.
        
        Returns:
            tuple: (waveform, sample_rate)
        """
        try:
            # Try torchaudio first 
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Check if resampling is needed
            if sample_rate != 16000:
                # TitaNet typically expects 16kHz
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
                
            return waveform, sample_rate
            
        except Exception:
            # Fall back to librosa
            import librosa
            import torch
            
            waveform, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
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