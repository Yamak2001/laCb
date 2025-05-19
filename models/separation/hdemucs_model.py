# models/separation/hdemucs_model.py
import os
import numpy as np
import torch
import librosa
import soundfile as sf
import time
import logging
import subprocess
from typing import List, Dict, Any, Tuple

from models.base import BaseSeparationModel

# Set up logging
logger = logging.getLogger(__name__)

# Directory setup
MODELS_DIR = "models/pretrained"
TEMP_DIR = "data/temp"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

class HDemucsModel(BaseSeparationModel):
    """Hybrid Demucs model for audio source separation (Facebook/Meta)"""
    def __init__(self, model_name: str, use_gpu: bool = False, sample_rate: int = 44100, 
                 model_variant: str = "hdemucs_mmi"):
        super().__init__(model_name, use_gpu)
        self.sample_rate = sample_rate  # HDemucs typically uses 44.1kHz
        self.device = None
        self.model = None
        self.model_variant = model_variant  # Options: hdemucs, hdemucs_mmi
        self.use_command_line = False  # Fallback to command line if Python API fails
    def _check_dependencies(self) -> List[str]:
        """Check for missing dependencies and return a list of them."""
        missing = []
        
        try:
            import torch
        except ImportError:
            missing.append("torch")
        
        try:
            import torchaudio
        except ImportError:
            missing.append("torchaudio")
            
        try:
            import yaml
        except ImportError:
            missing.append("pyyaml")
            
        try:
            import dora
        except ImportError:
            missing.append("dora-search")
            
        try:
            import julius
        except ImportError:
            missing.append("julius")
            
        # Check for demucs package
        try:
            import demucs
        except ImportError:
            missing.append("demucs")
            
        # Check if demucs is available as a command
        try:
            result = subprocess.run(["demucs", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                missing.append("demucs command")
        except:
            # It's not a missing dependency, just not available as command
            pass
            
        return missing
    def _install_dependencies(self) -> bool:
        """Attempt to install missing dependencies."""
        try:
            missing = self._check_dependencies()
            if missing:
                logger.warning(f"Missing dependencies: {', '.join(missing)}")
                logger.info("Attempting to install missing dependencies...")
                
                # Install core dependencies
                if "torch" in missing or "torchaudio" in missing:
                    subprocess.run(["pip", "install", "torch", "torchaudio"], check=True)
                
                if "pyyaml" in missing:
                    subprocess.run(["pip", "install", "pyyaml"], check=True)
                    
                if "dora-search" in missing:
                    subprocess.run(["pip", "install", "dora-search"], check=True)
                    
                if "julius" in missing:
                    subprocess.run(["pip", "install", "julius"], check=True)
                
                # Try to install demucs 
                if "demucs" in missing:
                    try:
                        # Try with specific version for hdemucs compatibility
                        subprocess.run(["pip", "install", "demucs>=4.0.0"], check=True)
                    except subprocess.CalledProcessError:
                        logger.warning("Failed to install demucs via pip")
                    
                return True
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            
        return False
    def load_model(self) -> None:
        """Load a pre-trained HDemucs model for voice separation."""
        start_time = time.time()
        
        try:
            # Check dependencies and try to install missing ones
            missing = self._check_dependencies()
            if missing and not self._install_dependencies():
                # If installation failed, suggest manual installation
                raise ImportError(f"Missing dependencies: {', '.join(missing)}. Please install them manually.")
            
            # Import dependencies
            try:
                import torch
                import torchaudio
                
                logger.info(f"Loading HDemucs model variant: {self.model_variant}")
                
                # Set device
                self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
                
                try:
                    # First try: import demucs
                    import demucs
                    from demucs.pretrained import get_model
                    model = get_model(self.model_variant)
                    logger.info("Loaded model using demucs.pretrained")
                except (ImportError, AttributeError, ValueError) as e1:
                    logger.warning(f"Could not load model using demucs.pretrained: {e1}")
                    try:
                        # Second try: torch.hub with specific repo
                        model = torch.hub.load('facebookresearch/demucs:main', self.model_variant)
                        logger.info("Loaded model using torch.hub with facebookresearch/demucs:main")
                    except Exception as e2:
                        logger.warning(f"Could not load model using torch.hub: {e2}")
                        try:
                            # Third try: torch.hub with newer repo
                            model = torch.hub.load('facebookresearch/demucs', self.model_variant)
                            logger.info("Loaded model using torch.hub with facebookresearch/demucs")
                        except Exception as e3:
                            logger.warning(f"Could not load any model: {e3}")
                            # Fall back to command line
                            logger.info("Will use command-line demucs as fallback")
                            self.use_command_line = True
                                                        return
                
                # Move model to device
                model.to(self.device)
                model.eval()
                
                # Store model
                self.model = model
                
                logger.info(f"HDemucs model loaded successfully on {self.device}")
                
            except Exception as e:
                logger.error(f"Error during HDemucs model loading: {e}")
                # Fallback to command line
                logger.info("Will use command-line demucs as fallback")
                self.use_command_line = True
                
        except Exception as e:
            logger.error(f"Failed to load HDemucs model: {e}")
            raise
        
        finally:
                def _separate_using_command_line(self, audio_path: str, output_dir: str) -> List[str]:
        """Use the demucs command-line tool as a fallback."""
        logger.info("Using command-line demucs for separation")
        
        # Generate a unique output directory for this operation
        timestamp = int(time.time())
        temp_output_dir = os.path.join(output_dir, f"demucs_output_{timestamp}")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Generate base output filename
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        try:
            # Run demucs command with the specified model
            cmd = [
                "demucs", 
                "--out", temp_output_dir,
                "--model", self.model_variant,
                audio_path
            ]
            
            if not self.use_gpu:
                cmd.append("--device")
                cmd.append("cpu")
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Demucs creates a directory structure like:
            # output_dir/model_name/track_name/stem.wav
            model_dir = os.path.join(temp_output_dir, self.model_variant)
            if not os.path.exists(model_dir):
                # Try to find any subdirectory
                subdirs = [d for d in os.listdir(temp_output_dir) if os.path.isdir(os.path.join(temp_output_dir, d))]
                if subdirs:
                    model_dir = os.path.join(temp_output_dir, subdirs[0])
                else:
                    raise FileNotFoundError(f"Could not find model directory in {temp_output_dir}")
            
            # Find the audio stem directory
            track_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
            if not track_dirs:
                raise FileNotFoundError(f"Could not find any track directories in {model_dir}")
            
            track_dir = os.path.join(model_dir, track_dirs[0])
            
            # Collect the stems
            output_paths = []
            # Typical stems are 'vocals', 'drums', 'bass', 'other'
            expected_stems = ['vocals', 'drums', 'bass', 'other']
            
            # Look for vocals first to ensure it's at the beginning of our list
            vocals_path = os.path.join(track_dir, 'vocals.wav')
            if os.path.exists(vocals_path):
                # Copy to our standard output format
                dest_path = os.path.join(output_dir, f"{base_filename}_vocals.wav")
                shutil.copy(vocals_path, dest_path)
                output_paths.append(dest_path)
            
            # Add other stems
            for stem in expected_stems:
                if stem == 'vocals':  # Already handled
                    continue
                    
                stem_path = os.path.join(track_dir, f'{stem}.wav')
                if os.path.exists(stem_path):
                    dest_path = os.path.join(output_dir, f"{base_filename}_{stem}.wav")
                    shutil.copy(stem_path, dest_path)
                    output_paths.append(dest_path)
            
            # Cleanup temporary directory
            try:
                import shutil
                shutil.rmtree(temp_output_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary directory: {cleanup_error}")
            
            if not output_paths:
                raise RuntimeError("No output files were created during separation")
                
            return output_paths
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with error: {e}")
            logger.error(f"STDOUT: {e.stdout.decode() if e.stdout else 'None'}")
            logger.error(f"STDERR: {e.stderr.decode() if e.stderr else 'None'}")
            raise RuntimeError(f"Command-line separation failed: {e}")
        except Exception as e:
            logger.error(f"Error in command-line separation: {e}")
            raise
        def separate_sources(self, audio_path: str, output_dir: str) -> List[str]:
        """Separate audio sources from a mixed input file using the HDemucs model."""
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate base output filename from input
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Check if file exists and is readable
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Ensure the model is loaded (or command line mode is set)
        if self.model is None and not self.use_command_line:
            logger.info("Model not loaded yet, loading now")
            self.load_model()
        
        # If using command line mode, delegate to that method
        if self.use_command_line:
            return self._separate_using_command_line(audio_path, output_dir)
        
        try:
            # Import needed libraries
            import torch
            import torchaudio
            
            # Load audio file
            logger.info(f"Loading audio file: {audio_path}")
            audio, orig_sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            target_sr = self.sample_rate
            if hasattr(self.model, 'samplerate'):
                target_sr = self.model.samplerate
            
            if orig_sr != target_sr:
                logger.info(f"Resampling from {orig_sr}Hz to {target_sr}Hz")
                resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
                audio = resampler(audio)
            
            # Convert to mono if necessary for speech models
            if audio.shape[0] > 1 and hasattr(self.model, 'audio_channels') and self.model.audio_channels == 1:
                logger.info(f"Converting {audio.shape[0]} channels to mono")
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Move to device
            audio = audio.to(self.device)
            
            # Run the model
            logger.info("Processing audio with HDemucs")
            with torch.no_grad():
                try:
                    # Try the separate method first
                    if hasattr(self.model, 'separate'):
                        logger.info("Using model.separate() method")
                        sources = self.model.separate(audio)
                    else:
                        # Fall back to direct call
                        logger.info("Using direct model call")
                        sources = self.model(audio)
                except Exception as e:
                    # Try alternative approach if the direct methods fail
                    logger.warning(f"Error with direct methods, trying alternative: {e}")
                    try:
                        logger.info("Trying demucs.apply.apply_model")
                        from demucs.apply import apply_model
                        sources = apply_model(self.model, audio, shifts=1, split=True, overlap=0.25)
                    except ImportError:
                        # If that fails, try direct processing again but differently
                        logger.warning("Could not import demucs.apply, trying raw model again")
                        sources = self.model(audio)
            
            # Get source names
            source_names = []
            if hasattr(self.model, 'sources'):
                source_names = self.model.sources
            elif hasattr(self.model, 'source_names'):
                source_names = self.model.source_names
            else:
                # Default source names if not available
                source_names = ['drums', 'bass', 'other', 'vocals']
            
            logger.info(f"Source names: {source_names}")
            
            # Save each source to a separate file
            output_paths = []
            
            # Handle different output dimensions and formats
            if isinstance(sources, dict):
                # Newer Demucs versions might return a dictionary
                logger.info("Processing dictionary output")
                for name, source in sources.items():
                    source_np = source.cpu().numpy()[0]  # Take first channel if multi-channel
                    
                    # Normalize to avoid clipping
                    if np.max(np.abs(source_np)) > 0:
                        source_np = source_np / np.max(np.abs(source_np)) * 0.9
                    
                    # Save to file
                    output_file = os.path.join(output_dir, f"{base_filename}_{name}.wav")
                    sample_rate = target_sr
                    sf.write(output_file, source_np, sample_rate)
                    output_paths.append(output_file)
                    
                    logger.info(f"Saved {name} to {output_file}")
            else:
                # Handle tensor output (traditional format)
                logger.info("Processing tensor output")
                # Check if it's a tuple with metadata
                if isinstance(sources, tuple) and len(sources) > 0:
                    # Extract just the separated sources tensor
                    sources = sources[0]
                
                # Get the right number of sources
                num_sources = min(len(source_names), sources.shape[0])
                
                for i in range(num_sources):
                    # Get source name if available
                    source_name = source_names[i] if i < len(source_names) else f"source_{i+1}"
                    source = sources[i:i+1]  # Keep dimension
                    
                    # Normalize to avoid clipping
                    max_val = torch.max(torch.abs(source) + 1e-8)
                    source = source / max_val * 0.9
                    
                    # Move to CPU and convert to numpy
                    source_np = source.cpu().numpy()[0]  # Take first channel if multi-channel
                    
                    # Save to file
                    output_file = os.path.join(output_dir, f"{base_filename}_{source_name}.wav")
                    sample_rate = target_sr
                    sf.write(output_file, source_np, sample_rate)
                    output_paths.append(output_file)
                    
                    logger.info(f"Saved {source_name} to {output_file}")
            
            # For voice isolation, we're most interested in the "vocals" source
            # Let's make sure it's first in the list if available
            for i, path in enumerate(output_paths):
                if "vocals" in path or "voice" in path:
                    if i > 0:
                        # Move vocals to first position
                        vocals_path = output_paths.pop(i)
                        output_paths.insert(0, vocals_path)
                        logger.info(f"Moved vocals to first position in output list")
                    break
            
            # If no output paths were created, raise an error
            if not output_paths:
                raise RuntimeError("No output files were created during separation")
                
            return output_paths
            
        except Exception as e:
            logger.error(f"Error in HDemucs audio source separation: {e}")
            
            # Try the command line fallback if Python API fails
            logger.info("Trying command line fallback...")
            try:
                self.use_command_line = True
                return self._separate_using_command_line(audio_path, output_dir)
            except Exception as fallback_error:
                logger.error(f"Command line fallback also failed: {fallback_error}")
                # Re-raise the original error
                raise e