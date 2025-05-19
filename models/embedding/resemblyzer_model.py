# models/embedding/resemblyzer_model.py
import os
import numpy as np
import time
import logging
import config
from typing import List, Dict, Any

from resemblyzer import VoiceEncoder, preprocess_wav
from models.base import BaseEmbeddingModel

# Set up logging
logger = logging.getLogger(__name__)

class ResemblyzerModel(BaseEmbeddingModel):
    """Resemblyzer model for voice embedding creation and comparison"""
    
    def __init__(self, model_name: str, use_gpu: bool = False):
        super().__init__(model_name, use_gpu)
        self.model = None
    
    def load_model(self) -> None:
        """Load the Resemblyzer Voice Encoder model."""
        start_time = time.time()
        try:
            logger.info("Loading Resemblyzer Voice Encoder model")
            self.model = VoiceEncoder()
            logger.info("Resemblyzer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Resemblyzer model: {e}")
            raise
        
    
    def create_embedding(self, audio_path: str) -> np.ndarray:
        """Create a voice embedding from an audio file using Resemblyzer."""
        # Ensure the model is loaded
        if self.model is None:
            logger.info("Model not loaded yet, loading now")
            self.load_model()
        
        try:
            logger.info(f"Creating voice embedding from: {audio_path}")
            
            # Check if file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Preprocess the audio file
            wav = preprocess_wav(audio_path)
            
            # Create embedding
            embedding = self.model.embed_utterance(wav)
            
            logger.info(f"Created embedding with shape: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating voice embedding: {e}")
            raise
    
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