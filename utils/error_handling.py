# utils/error_handling.py
import logging
import traceback
import functools
import os
import numpy as np
from typing import Callable, List, Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def safe_operation(fallback_value=None, log_exceptions=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_exceptions:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(traceback.format_exc())
                return fallback_value
        return wrapper
    return decorator

class ModelFailure(Exception):
    def __init__(self, message: str, model_id: str, alternative_models: List[str] = None):
        super().__init__(message)
        self.model_id = model_id
        self.alternative_models = alternative_models or []
        
    def __str__(self):
        msg = f"Model '{self.model_id}' failed: {super().__str__()}"
        if self.alternative_models:
            msg += f"\nTry these alternative models: {', '.join(self.alternative_models)}"
        return msg

def get_fallback_models(model_type: str) -> Dict[str, str]:
    fallbacks = {
        'separation': {
            'sepformer': 'convtasnet',
            'dprnn': 'convtasnet',
            'hdemucs': 'demucs',
            'nemo': 'convtasnet',
        },
        'embedding': {
            'ecapa': 'resemblyzer',
            'speechbrain': 'resemblyzer',
            'pyannote': 'resemblyzer',
            'titanet': 'resemblyzer',
        },
        'vad': {
            'pyannote_vad': 'webrtcvad',
            'speechbrain_vad': 'webrtcvad',
        }
    }
    
    return fallbacks.get(model_type, {})

def retry_with_fallback(max_retries=2, fallback_function=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1}/{max_retries} failed in {func.__name__}: {e}")
                    if attempt == max_retries - 1:
                        if fallback_function:
                            logger.info(f"Using fallback function for {func.__name__}")
                            return fallback_function(*args, **kwargs)
                        raise
            raise RuntimeError(f"All attempts failed in {func.__name__}")
        return wrapper
    return decorator

def handle_audio_file_error(audio_path: str) -> None:
    # Check if file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Check file size
    file_size = os.path.getsize(audio_path)
    if file_size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    
    # Check file extension
    ext = os.path.splitext(audio_path)[1].lower()
    supported_exts = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.webm']
    if ext not in supported_exts:
        logger.warning(f"File extension '{ext}' might not be supported. Proceed with caution.")

def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray, eps=1e-8) -> float:
    # Check inputs
    if embedding1 is None or embedding2 is None:
        logger.error("Cannot compute similarity on None embeddings")
        return 0.0
    
    # Squeeze any extra dimensions
    embedding1 = embedding1.squeeze()
    embedding2 = embedding2.squeeze()
    
    # Check shapes
    if embedding1.shape != embedding2.shape:
        logger.warning(f"Embedding shapes don't match: {embedding1.shape} vs {embedding2.shape}")
        
        # Handle common dimension mismatches
        dim1, dim2 = embedding1.shape[0], embedding2.shape[0]
        
        if dim1 == 256 and dim2 == 512:
            # Resemblyzer vs SpeechBrain - project to common dimension
            logger.info("Projecting embeddings to common dimension")
            # Simple dimensionality reduction by taking first 256 dimensions
            embedding2 = embedding2[:256]
        elif dim1 == 512 and dim2 == 256:
            # SpeechBrain vs Resemblyzer
            logger.info("Projecting embeddings to common dimension")
            embedding1 = embedding1[:256]
        elif dim1 < dim2:
            # General case - truncate longer embedding
            logger.info(f"Truncating embedding from {dim2} to {dim1}")
            embedding2 = embedding2[:dim1]
        else:
            # Truncate the other way
            logger.info(f"Truncating embedding from {dim1} to {dim2}")
            embedding1 = embedding1[:dim2]
        
        # Final check
        if embedding1.shape != embedding2.shape:
            logger.error("Still cannot match embedding dimensions")
            return 0.0
    
    try:
        # Compute cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        # Avoid division by zero
        if norm1 < eps or norm2 < eps:
            logger.warning("Near-zero norm in embeddings, returning zero similarity")
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure the result is valid
        if np.isnan(similarity) or np.isinf(similarity):
            logger.warning(f"Invalid similarity value: {similarity}, returning zero")
            return 0.0
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0

def robust_model_selection(preferred_model_id: str, available_models: List[str], model_type: str) -> str:
    # First try the preferred model
    if preferred_model_id in available_models:
        return preferred_model_id
    
    # Try fallbacks for this model
    fallbacks = get_fallback_models(model_type)
    if preferred_model_id in fallbacks and fallbacks[preferred_model_id] in available_models:
        logger.info(f"Using fallback model {fallbacks[preferred_model_id]} instead of {preferred_model_id}")
        return fallbacks[preferred_model_id]
    
    # If no fallbacks found, use the first available model
    if available_models:
        logger.warning(f"Model {preferred_model_id} not available. Using {available_models[0]} instead.")
        return available_models[0]
    
    # No models available
    raise ValueError(f"No {model_type} models available")
