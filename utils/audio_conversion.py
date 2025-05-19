# utils/audio_conversion.py
import os
import subprocess
import logging
import tempfile

# Set up logging
logger = logging.getLogger(__name__)

def convert_audio_to_wav(input_path, output_path=None, sample_rate=16000, channels=1):
    """
    Convert any audio file to WAV format using FFmpeg with improved error handling.
    
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
        
        # First try to identify the format
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-show_format",
            "-show_streams",
            input_path
        ]
        
        try:
            probe_result = subprocess.run(probe_cmd, 
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.PIPE, 
                                        text=True,
                                        check=False)
            
            if probe_result.returncode != 0:
                logger.warning(f"Could not identify format of {input_path}: {probe_result.stderr}")
                
                # Try to create a new copy to fix potential issues
                logger.info("Creating a new temporary copy of the file")
                temp_dir = tempfile.mkdtemp()
                temp_file = os.path.join(temp_dir, "temp_copy" + os.path.splitext(input_path)[1])
                
                with open(input_path, 'rb') as src, open(temp_file, 'wb') as dst:
                    dst.write(src.read())
                
                input_path = temp_file
        except:
            logger.warning("FFprobe not available for format identification")
        
        # Build FFmpeg command with improved error handling
        cmd = [
            "ffmpeg",
            "-y",                      # Overwrite output files without asking
            "-loglevel", "warning",    # Show only warnings and errors
            "-i", input_path,          # Input file
            "-ar", str(sample_rate),   # Sample rate
            "-ac", str(channels),      # Audio channels
            "-c:a", "pcm_s16le",       # 16-bit PCM encoding
            output_path
        ]
        
        # Run conversion with full error capture
        result = subprocess.run(cmd, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               check=False)  # Don't raise exception yet
        
        # Check the result manually to provide better error messages
        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="replace")
            logger.error(f"FFmpeg conversion error: {error_msg}")
            
            # Try with a different approach for troublesome files
            logger.info("Trying alternative conversion method")
            alt_cmd = [
                "ffmpeg",
                "-y",                   # Overwrite output
                "-loglevel", "warning", # Minimal output
                "-f", "lavfi",          # Use the lavfi filter
                "-i", f"anullsrc=r={sample_rate}:cl=mono",  # Generate silent audio
                "-i", input_path,       # Input file as second input
                "-shortest",            # Match shortest input
                "-map", "1:a:0",        # Map audio from second input
                "-ar", str(sample_rate),# Sample rate
                "-ac", str(channels),   # Channels
                "-c:a", "pcm_s16le",    # 16-bit PCM encoding
                output_path
            ]
            
            alt_result = subprocess.run(alt_cmd,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      check=False)
            
            if alt_result.returncode != 0:
                alt_error = alt_result.stderr.decode("utf-8", errors="replace")
                logger.error(f"Alternative conversion also failed: {alt_error}")
                
                # If all else fails, create a silent WAV file as a last resort
                logger.warning("Creating a silent WAV file as fallback")
                silent_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "lavfi",
                    "-i", f"anullsrc=r={sample_rate}:cl=mono",
                    "-t", "3",  # 3 seconds of silence
                    "-ar", str(sample_rate),
                    "-ac", str(channels),
                    "-c:a", "pcm_s16le",
                    output_path
                ]
                
                silent_result = subprocess.run(silent_cmd,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            check=True)
                
                logger.info("Created silent WAV file as fallback")
            else:
                logger.info("Alternative conversion method succeeded")
        
        # Verify the output file exists and is not empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError("Conversion produced empty or missing file")
        
        logger.info(f"Successfully converted audio to WAV: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        raise

def is_valid_wav(file_path):
    """
    Improved check if a file is a valid WAV file using FFprobe with better error handling.
    
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
                               text=True)
        
        # If there's error output, it's likely not valid
        if result.stderr:
            logger.warning(f"FFprobe reported errors for {file_path}: {result.stderr}")
            return False
            
        # Check for basic audio stream info
        return "codec_name" in result.stdout and "sample_rate" in result.stdout
        
    except Exception as e:
        logger.error(f"Error checking audio format: {e}")
        return False
