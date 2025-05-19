# models/base.py
from abc import ABC, abstractmethod
import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional


class BaseSeparationModel(ABC):
    """Abstract base class for all audio source separation models."""
    
    def __init__(self, model_name: str, use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = None
        self.model = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def separate_sources(self, audio_path: str, output_dir: str) -> List[str]:
        """
        Separate audio sources from a mixed input file.
        
        Args:
            audio_path (str): Path to the mixed audio file
            output_dir (str): Directory to save separated sources
            
        Returns:
            List[str]: List of paths to separated source audio files
        """
        pass


class BaseEmbeddingModel(ABC):
    """Abstract base class for all voice embedding models."""
    
    def __init__(self, model_name: str, use_gpu: bool = False):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = None
        self.model = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def create_embedding(self, audio_path: str) -> np.ndarray:
        """
        Create a voice embedding from an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Voice embedding vector
        """
        pass


class BaseVADProcessor(ABC):
    """Abstract base class for Voice Activity Detection processors."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the VAD model."""
        pass
    
    @abstractmethod
    def process_audio(self, input_path: str, output_path: str) -> str:
        """
        Process audio to detect speech segments.
        
        Args:
            input_path (str): Path to input audio file
            output_path (str): Path to save processed audio
            
        Returns:
            str: Path to the processed audio file
        """
        pass