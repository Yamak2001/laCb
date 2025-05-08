import os
import numpy as np
import torch
import librosa
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from asteroid.models import ConvTasNet
import time
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Directory setup
MODELS_DIR = "models"
TEMP_DIR = "data/temp"
ISOLATED_DIR = "data/isolated"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(ISOLATED_DIR, exist_ok=True)

def load_conv_tasnet_model(use_gpu=False, max_retries=3):
    """Load a pre-trained Conv-TasNet model for voice separation with retry logic."""
    
    # Try multiple model sources in case one fails
    model_sources = [
        "conv_tasnet_sepnoisy_libri2Mix_enhsingle_16k",  # Standard Asteroid model name
        "JorisCos/ConvTasNet_Libri2Mix_sepnoisy_16k",    # Hugging Face model path
        "asteroidevaluate/ConvTasNet_Libri2Mix_sepnoisy_16k"  # Alternative source
    ]
    
    # Try loading from each source with retries
    for source in model_sources:
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to load model from {source} (attempt {attempt+1}/{max_retries})")
                model = ConvTasNet.from_pretrained(
                    source,
                    cache_dir=MODELS_DIR
                )
                logger.info("Model loaded successfully!")
                
                # Move model to GPU if available and requested
                device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
                if device.type == "cuda":
                    logger.info("Using GPU acceleration")
                else:
                    logger.info("Using CPU for processing")
                    
                model.to(device)
                model.eval()  # Set model to evaluation mode
                
                return model, device
                
            except Exception as e:
                logger.error(f"Failed to load model from {source}: {e}")
                if attempt < max_retries - 1:
                    # Wait before retrying
                    wait_time = 2 * (attempt + 1)  # Exponential backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
    
    # If all sources and retries fail, create a fallback model
    logger.warning("All model loading attempts failed. Creating fallback model.")
    model = ConvTasNet(
        n_src=2,  # Number of sources to separate
        n_repeats=3,  # Number of repeats
        n_filters=512,  # Number of filters in autoencoder
        kernel_size=16,  # Kernel size in convolutional blocks
        n_blocks=8,  # Number of convolutional blocks in each repeat
        n_channels=128,  # Number of channels in bottleneck
        norm_type="gLN"  # Normalization type
    )
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    logger.info("Created fallback model")
    return model, device

def separate_audio_sources(model, mixed_audio_path, output_dir=TEMP_DIR, sample_rate=16000, device="cpu"):
    """Separate audio sources from a mixed input file using the Conv-TasNet model."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base output filename from input
    base_filename = os.path.splitext(os.path.basename(mixed_audio_path))[0]
    
    # Check if file exists and is readable
    if not os.path.exists(mixed_audio_path):
        raise FileNotFoundError(f"Audio file not found: {mixed_audio_path}")
    
    try:
        # Load and resample audio if necessary
        logger.info(f"Loading audio file: {mixed_audio_path}")
        audio, orig_sr = librosa.load(mixed_audio_path, sr=None, mono=True)
        logger.info(f"Audio loaded successfully. Original sample rate: {orig_sr}Hz")
        
        # Resample if necessary
        if orig_sr != sample_rate:
            logger.info(f"Resampling from {orig_sr}Hz to {sample_rate}Hz")
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sample_rate)
        
        # Normalize audio to prevent clipping
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Log audio properties
        logger.info(f"Audio duration: {len(audio)/sample_rate:.2f} seconds")
        logger.info(f"Audio shape: {audio.shape}")
        
        # Convert to tensor and reshape for model
        logger.info("Processing audio with model")
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Separate sources
        with torch.no_grad():
            model_output = model(audio_tensor)
        
        # Convert output to numpy array
        # Model output shape will be [1, n_sources, audio_length]
        separated = model_output.cpu().numpy()
        
        # Log separated output properties
        logger.info(f"Separated output shape: {separated.shape}")
        
        # Save separated sources
        output_paths = []
        for idx in range(separated.shape[1]):
            source = separated[0, idx]
            output_file = os.path.join(output_dir, f"{base_filename}_source_{idx+1}.wav")
            
            # Normalize output
            source = source / np.max(np.abs(source)) * 0.9
            
            # Save to file
            sf.write(output_file, source, sample_rate)
            output_paths.append(output_file)
            logger.info(f"Saved source {idx+1} to {output_file}")
        
        return output_paths
        
    except Exception as e:
        logger.error(f"Error in audio source separation: {e}")
        raise

def match_voice_to_embedding(source_paths, target_embedding):
    """Match separated voice sources to target voice embedding."""
    logger.info("Loading voice encoder model")
    encoder = VoiceEncoder()
    similarities = []
    
    logger.info(f"Found {len(source_paths)} source paths to evaluate")
    for i, path in enumerate(source_paths):
        try:
            logger.info(f"Processing source {i+1}: {path}")
            # Preprocess audio for Resemblyzer
            wav = preprocess_wav(path)
            
            # Extract embedding
            source_embedding = encoder.embed_utterance(wav)
            
            # Calculate cosine similarity
            similarity = np.dot(source_embedding, target_embedding) / (
                np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding)
            )
            similarities.append(similarity)
            logger.info(f"Source {i+1} similarity: {similarity:.4f}")
            
        except Exception as e:
            logger.error(f"Error processing source {i+1}: {e}")
            similarities.append(-1)  # Mark as invalid
    
    if not similarities or max(similarities) < 0:
        raise ValueError("Could not match any voice sources")
    
    # Find the index of the most similar source
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    logger.info(f"Best match: Source {best_match_idx+1} with similarity {best_similarity:.4f}")
    return best_match_idx, best_similarity

def isolate_voice(mixed_audio_path, profile_id, db_session, output_dir=ISOLATED_DIR, use_gpu=False):
    """Complete function to isolate a specific voice from mixed audio."""
    logger.info(f"Starting voice isolation for profile ID {profile_id}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Load voice profile embedding
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
        
        # Step 2: Load the separation model
        logger.info("Loading separation model")
        model, device = load_conv_tasnet_model(use_gpu=use_gpu)
        
        # Step 3: Separate audio sources
        logger.info(f"Separating audio sources from {mixed_audio_path}")
        separated_paths = separate_audio_sources(
            model,
            mixed_audio_path,
            TEMP_DIR,
            device=device
        )
        
        # Step 4: Find best matching source
        logger.info("Finding best matching voice")
        best_match_idx, similarity = match_voice_to_embedding(separated_paths, target_embedding)
        
        # Step 5: Create output file path with timestamp for uniqueness
        timestamp = int(time.time())
        output_filename = f"{profile.name}_isolated_{timestamp}.wav"
        isolated_path = os.path.join(output_dir, output_filename)
        
        # Copy the best matching source to the output path
        logger.info(f"Saving isolated voice to {isolated_path}")
        shutil.copy(separated_paths[best_match_idx], isolated_path)
        
        logger.info("Voice isolation complete")
        return isolated_path, similarity
        
    except Exception as e:
        logger.error(f"Error in voice isolation: {e}")
        # Re-raise the exception with more context
        raise Exception(f"Voice isolation failed: {str(e)}")