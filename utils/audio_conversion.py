# Add this to utils/audio_conversion.py

import os
import subprocess
import logging

# Set up logging
logger = logging.getLogger(__name__)

def convert_audio_to_wav(input_path, output_path=None, sample_rate=16000, channels=1):
    """
    Convert any audio file to WAV format using FFmpeg.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str, optional): Path for output WAV file. If None, replaces extension with .wav
        sample_rate (int): Target sample rate in Hz
        channels (int): Number of audio channels (1=mono, 2=stereo)
        
    Returns:
        str: Path to the converted WAV file
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Generate output path if not provided
    if not output_path:
        output_path = os.path.splitext(input_path)[0] + ".wav"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        logger.info(f"Converting {input_path} to WAV format at {output_path}")
        
        # Check if FFmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE, 
                          check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("FFmpeg is not installed or not in PATH")
        
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ar", str(sample_rate),   # Sample rate
            "-ac", str(channels),      # Audio channels
            "-c:a", "pcm_s16le",       # 16-bit PCM encoding
            "-y",                      # Overwrite output
            output_path
        ]
        
        # Run conversion
        result = subprocess.run(cmd, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=True)
        
        # Verify the output file exists and is not empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Conversion produced empty or missing file")
        
        logger.info(f"Successfully converted audio to WAV: {output_path}")
        return output_path
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8") if e.stderr else str(e)
        logger.error(f"FFmpeg conversion error: {error_msg}")
        raise RuntimeError(f"Audio conversion failed: {error_msg}")
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        raise

def is_valid_wav(file_path):
    """
    Check if a file is a valid WAV file by attempting to get its properties with FFprobe.
    
    Args:
        file_path (str): Path to the audio file to check
        
    Returns:
        bool: True if file is a valid WAV, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        # Use FFprobe to get audio stream info
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-of", "default=noprint_wrappers=1",
            file_path
        ]
        
        result = subprocess.run(cmd, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True,
                               check=True)
        
        # If we get valid output, it's a valid audio file
        return "codec_name" in result.stdout and "sample_rate" in result.stdout
        
    except Exception:
        return False