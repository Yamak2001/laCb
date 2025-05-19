# utils/audio_processing.py
import os
import numpy as np
import torch
import time
import shutil
import logging
import config
from typing import Tuple, Optional, Dict, Any, List

# Import model factories
from models.factory import (
    SeparationModelFactory, 
    EmbeddingModelFactory, 
    VADProcessorFactory,
    register_all_models
)

# Import error handling utilities
from utils.error_handling import (
    safe_operation,
    retry_with_fallback,
    handle_audio_file_error,
    robust_model_selection,
    compute_similarity
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory setup
MODELS_DIR = config.PRETRAINED_DIR
TEMP_DIR = config.TEMP_DIR
ISOLATED_DIR = config.ISOLATED_DIR
# Directories created by config
# Directories created by config
# Directories created by config

# Initialize model registrations
register_all_models()

@retry_with_fallback(max_retries=2)
def match_voice_to_embedding(
    source_paths: List[str], 
    target_embedding: np.ndarray,
    embedding_model_id: str = "resemblyzer",
    use_gpu: bool = False,
    profile_embedding_model_id: str = None
) -> Tuple[int, float]:
    """
    Match separated voice sources to target voice embedding.
    Now handles different embedding models between profile and current process.
    
    Args:
        source_paths: List of paths to separated audio sources
        target_embedding: Target voice embedding to match
        embedding_model_id: ID of the embedding model to use
        use_gpu: Whether to use GPU for processing
        
    Returns:
        Tuple containing the index of the best match and its similarity score
    """
    # Check inputs
    if not source_paths:
        raise ValueError("No source paths provided")
    
    if target_embedding is None:
        raise ValueError("Target embedding is None")
    
    # Get available embedding models
    available_models = list(EmbeddingModelFactory.list_available_models().keys())
    
    # Select model robustly
    model_id = robust_model_selection(embedding_model_id, available_models, "embedding")
    
    # Get the embedding model
    embedding_model = EmbeddingModelFactory.get_model(
        model_id,
        use_gpu=use_gpu
    )
    
    # Load the model
    if not hasattr(embedding_model, 'model') or embedding_model.model is None:
        embedding_model.load_model()
    
    similarities = []
    error_count = 0
    
    logger.info(f"Found {len(source_paths)} source paths to evaluate")
    for i, path in enumerate(source_paths):
        try:
            logger.info(f"Processing source {i+1}: {path}")
            
            # Verify the file exists
            handle_audio_file_error(path)
            
            # Create embedding for the source
            source_embedding = embedding_model.create_embedding(path)
            
            # Calculate similarity safely
            similarity = compute_similarity(source_embedding, target_embedding)
            similarities.append(similarity)
            logger.info(f"Source {i+1} similarity: {similarity:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing source {i+1}: {e}")
            similarities.append(-1)  # Mark as invalid
            error_count += 1
    
    # Check if all sources failed
    if error_count == len(source_paths):
        raise ValueError("Failed to process all voice sources")
    
    if not similarities or max(similarities) < 0:
        raise ValueError("Could not match any voice sources")
    
    # Find the index of the most similar source
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    logger.info(f"Best match: Source {best_match_idx+1} with similarity {best_similarity:.4f}")
    return best_match_idx, best_similarity

def fallback_match_voice(*args, **kwargs):
    """Fallback function for voice matching using basic rules.
    Takes variable args to be compatible with retry_with_fallback decorator.
    """
    # Extract only the required arguments
    if len(args) >= 2:
        source_paths = args[0]
        target_embedding = args[1]
    else:
        # Try to get from keyword arguments
        source_paths = kwargs.get('source_paths', [])
        target_embedding = kwargs.get('target_embedding', None)
        
    logger.info("Using fallback voice matching based on simple rules")
    logger.info("Using fallback voice matching based on simple rules")
    
    # If source_paths is empty or None, return a default
    if not source_paths:
        logger.warning("No source paths provided to fallback function")
        return 0, 0.1  # Return first source with low confidence
    
    # First, check if there's a 'vocals' or 'voice' file
    for i, path in enumerate(source_paths):
        if 'vocals' in path.lower() or 'voice' in path.lower():
            logger.info(f"Selected source {i+1} based on filename")
            return i, 0.5  # Moderate confidence
    
    # If not, select the source with the longest duration (often the main voice)
    try:
        import librosa
        durations = []
        for path in source_paths:
            try:
                y, sr = librosa.load(path, sr=None)
                durations.append(len(y) / sr)
            except Exception as e:
                logger.warning(f"Error analyzing duration of {path}: {e}")
                durations.append(0)
        
        if durations:
            best_idx = np.argmax(durations)
            logger.info(f"Selected source {best_idx+1} based on duration")
            return best_idx, 0.4  # Lower confidence
    except ImportError:
        logger.warning("Librosa not available for duration analysis")
    except Exception as e:
        logger.error(f"Error in duration-based selection: {e}")
    
    # Last resort: just pick the first source
    logger.info("Selected first source as fallback")
    return 0, 0.3  # Low confidence


# Fallback function for isolate_voice
def fallback_isolate_voice(
    mixed_audio_path: str, 
    profile_id: int, 
    db_session, 
    output_dir: str = ISOLATED_DIR,
    use_gpu: bool = False,
    separation_model_id: str = "convtasnet",
    embedding_model_id: str = "resemblyzer",
    use_vad: bool = False,
    vad_processor_id: str = "webrtcvad",
    return_metrics: bool = False
) -> Tuple[str, float, Optional[Dict[str, Any]]]:
    """Fallback function for voice isolation when normal processing fails"""
    logger.warning("Using fallback voice isolation method")
    
    # Create a simple fallback that just copies the input file
    timestamp = int(time.time())
    
    try:
        # Get profile name
        from models.voice_profile import VoiceProfile
        profile = db_session.query(VoiceProfile).get(profile_id)
        if profile:
            output_filename = f"{profile.name}_isolated_fallback_{timestamp}.wav"
        else:
            output_filename = f"fallback_isolated_{timestamp}.wav"
            
        output_path = os.path.join(output_dir, output_filename)
        
        # Simply copy the mixed audio as a fallback
        shutil.copy(mixed_audio_path, output_path)
        
        # Return with low confidence
        return output_path, similarity, None
    except Exception as e:
        logger.error(f"Fallback also failed: {e}")
        raise

@retry_with_fallback(max_retries=2, fallback_function=fallback_isolate_voice)
def isolate_voice(
    mixed_audio_path: str, 
    profile_id: int, 
    db_session, 
    output_dir: str = ISOLATED_DIR,
    use_gpu: bool = False,
    separation_model_id: str = "convtasnet",
    embedding_model_id: str = "resemblyzer",
    use_vad: bool = False,
    vad_processor_id: str = "webrtcvad",
    return_metrics: bool = False
) -> Tuple[str, float, Optional[Dict[str, Any]]]:
    """
    Complete function to isolate a specific voice from mixed audio.
    
    Args:
        mixed_audio_path: Path to mixed audio file
        profile_id: ID of the voice profile to isolate
        db_session: Database session
        output_dir: Directory to save isolated voice
        use_gpu: Whether to use GPU for processing
        separation_model_id: ID of the separation model to use
        embedding_model_id: ID of the embedding model to use
        use_vad: Whether to use Voice Activity Detection
        vad_processor_id: ID of the VAD processor to use
        return_metrics: Whether to return performance metrics
        
    Returns:
        Tuple containing path to isolated voice, similarity score, and optional metrics
    """
    logger.info(f"Starting voice isolation for profile ID {profile_id}")
    logger.info(f"Using separation model: {separation_model_id}")
    logger.info(f"Using embedding model: {embedding_model_id}")
    
    # Record start time for overall processing
    start_time = time.time()
    
    # Initialize metrics dictionary if requested
    metrics = None
    if return_metrics:
        metrics = {
            "start_time": start_time,
            "models": {
                "separation": separation_model_id,
                "embedding": embedding_model_id,
                "vad": vad_processor_id if use_vad else None
            },
            "timings": {},
            "errors": [],
            "use_gpu": use_gpu
        }
        
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Load voice profile embedding
        profile_start = time.time()
        from models.voice_profile import VoiceProfile
        profile = db_session.query(VoiceProfile).get(profile_id)
        
        if not profile:
            raise ValueError(f"Voice profile with ID {profile_id} not found")
        
        logger.info(f"Loaded profile for '{profile.name}'")
        embedding_path = profile.embedding_path
        
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
            
        logger.info(f"Loading embedding from {embedding_path}")
        target_embedding = np.load(embedding_path)
                
        # Step 2: Apply VAD if requested
        processed_audio_path = mixed_audio_path
        if use_vad:
            vad_start = time.time()
            logger.info(f"Applying Voice Activity Detection using {vad_processor_id}")
            
            # Get available VAD processors
            available_vad = list(VADProcessorFactory.list_available_processors().keys())
            
            # Select processor robustly
            selected_vad = robust_model_selection(vad_processor_id, available_vad, "vad")
            
            # Get VAD processor
            vad_processor = VADProcessorFactory.get_processor(selected_vad)
            
            try:
                # Process audio with VAD
                vad_output_path = os.path.join(TEMP_DIR, f"vad_{os.path.basename(mixed_audio_path)}")
                processed_audio_path = vad_processor.process_audio(mixed_audio_path, vad_output_path)
                logger.info(f"VAD processed audio saved to: {processed_audio_path}")
            except Exception as e:
                logger.error(f"VAD processing failed: {e}")
                if return_metrics and metrics:
                    metrics["errors"].append(f"VAD error: {str(e)}")
                # Continue with original audio
                processed_audio_path = mixed_audio_path
        
        # Step 3: Load the separation model
        separation_start = time.time()
        logger.info(f"Loading separation model: {separation_model_id}")
        
        # Get available separation models
        available_separation = list(SeparationModelFactory.list_available_models().keys())
        
        # Select model robustly
        selected_separation = robust_model_selection(separation_model_id, available_separation, "separation")
        
        # Get separation model from factory
        separation_model = SeparationModelFactory.get_model(
            selected_separation,
            use_gpu=use_gpu
        )
        
        # Separate audio sources
        logger.info(f"Separating audio sources from {processed_audio_path}")
        separated_paths = separation_model.separate_sources(
            processed_audio_path,
            TEMP_DIR
        )
        
        # Save metrics
                                
        # Step 4: Find best matching source
        matching_start = time.time()
        logger.info("Finding best matching voice")
        best_match_idx, similarity = match_voice_to_embedding(
            separated_paths, 
            target_embedding,
            embedding_model_id,
            use_gpu
        )
        
                
        # Step 5: Create output file path with timestamp for uniqueness
        timestamp = int(time.time())
        output_filename = f"{profile.name}_isolated_{timestamp}.wav"
        isolated_path = os.path.join(output_dir, output_filename)
        
        # Copy the best matching source to the output path
        logger.info(f"Saving isolated voice to {isolated_path}")
        shutil.copy(separated_paths[best_match_idx], isolated_path)
        
        # Calculate overall processing time
        total_time = time.time() - start_time
        
        # Update metrics if requested
        if return_metrics and metrics:
            metrics["timings"]["total"] = total_time
            metrics["result"] = {
                "isolated_path": isolated_path,
                "similarity": float(similarity),
                "best_source_index": best_match_idx
            }
                        
        logger.info("Voice isolation complete")
        return isolated_path, similarity, metrics
        
    except Exception as e:
        logger.error(f"Error in voice isolation: {e}")
        # Add error to metrics if they're requested
        if return_metrics:
            metrics["errors"].append(str(e))
                                
        # Re-raise the exception with more context
        raise Exception(f"Voice isolation failed: {str(e)}")

def get_available_models():
    """
    Get all available models for the frontend.
    
    Returns:
        Dict containing available models and processors
    """
    # Initialize model registrations if not already done
    register_all_models()
    
    # Get models with their descriptive names
    separation_models = SeparationModelFactory.list_available_models()
    embedding_models = EmbeddingModelFactory.list_available_models()
    vad_processors = VADProcessorFactory.list_available_processors()
    
    # Add descriptions
    model_descriptions = {
        "separation_models": {
            "convtasnet": "ConvTasNet - Fast general-purpose speech separation",
            "dprnn": "DPRNN - Dual-Path RNN for speech separation",
            "demucs": "Demucs - Facebook AI music source separation",
            "sepformer": "SepFormer - State-of-the-art separation transformer",
            "hdemucs": "Hybrid Demucs - Enhanced hybrid music separation"
        },
        "embedding_models": {
            "resemblyzer": "Resemblyzer - Lightweight voice embedding",
            "speechbrain": "SpeechBrain X-Vector - Speaker recognition",
            "ecapa": "ECAPA-TDNN - Enhanced speaker embedding",
            "pyannote": "Pyannote Embedding - Speaker diarization model",
            "titanet": "TitaNet - NVIDIA speaker recognition model"
        },
        "vad_processors": {
            "webrtcvad": "WebRTC VAD - Fast voice activity detection",
            "speechbrain_vad": "SpeechBrain VAD - Neural voice activity detection",
            "pyannote_vad": "Pyannote VAD - State-of-the-art voice detection"
        }
    }
    
    # Combine available models with descriptions
    result = {
        "separation_models": {},
        "embedding_models": {},
        "vad_processors": {}
    }
    
    for model_id in separation_models:
        result["separation_models"][model_id] = model_descriptions["separation_models"].get(
            model_id, separation_models[model_id]
        )
    
    for model_id in embedding_models:
        result["embedding_models"][model_id] = model_descriptions["embedding_models"].get(
            model_id, embedding_models[model_id]
        )
    
    for proc_id in vad_processors:
        result["vad_processors"][proc_id] = model_descriptions["vad_processors"].get(
            proc_id, vad_processors[proc_id]
        )
    
    return result