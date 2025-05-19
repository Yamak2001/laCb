# models/vad/webrtcvad_processor.py
import os
import numpy as np
import soundfile as sf
import librosa
import time
import logging
import config
import collections
import webrtcvad
import struct
from typing import List, Dict, Any

from models.base import BaseVADProcessor

# Set up logging
logger = logging.getLogger(__name__)

class WebRtcVADProcessor(BaseVADProcessor):
    """WebRTC Voice Activity Detection processor."""
    
    def __init__(self, model_name: str, aggressiveness: int = 3, frame_duration_ms: int = 30):
        super().__init__(model_name)
        self.aggressiveness = min(max(0, aggressiveness), 3)  # 0-3, higher is more aggressive
        self.frame_duration_ms = frame_duration_ms  # 10, 20, or 30 ms
        self.sample_rate = config.SAMPLE_RATE  # WebRTC VAD only supports 8000, 16000, 32000, 48000
        self.model = None
    
    def load_model(self) -> None:
        """Load the WebRTC VAD model."""
        try:
            logger.info(f"Initializing WebRTC VAD with aggressiveness {self.aggressiveness}")
            self.model = webrtcvad.Vad(self.aggressiveness)
            logger.info("WebRTC VAD initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebRTC VAD: {e}")
            raise
    
    def frame_generator(self, audio: np.ndarray, frame_duration_ms: int, sample_rate: int):
        """Generate audio frames from numpy array."""
        # Calculate frame size in samples
        frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
        
        # Pad audio if needed to ensure complete frames
        if len(audio) % frame_size != 0:
            pad_length = frame_size - (len(audio) % frame_size)
            audio = np.pad(audio, (0, pad_length), 'constant')
        
        # Generate frames
        offset = 0
        while offset + frame_size <= len(audio):
            frame = audio[offset:offset + frame_size]
            # Convert to 16-bit PCM
            frame_bytes = struct.pack("%dh" % len(frame), *(np.int16(frame * 32767).tolist()))
            yield frame_bytes
            offset += frame_size
    
    def vad_collector(self, sample_rate: int, frame_duration_ms: int, padding_duration_ms: int, frames):
        """Filters out non-voiced audio frames."""
        num_padding_frames = padding_duration_ms // frame_duration_ms
        # Buffer for triggered audio frames
        triggered = False
        voiced_frames = []
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        
        for frame in frames:
            is_speech = self.model.is_speech(frame, sample_rate)
            
            if not triggered:
                ring_buffer.append((frame, is_speech))
                # If enough speech detected, start collecting
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    # Add all frames in the buffer
                    for f, _ in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # Keep adding speech frames
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                # If enough non-speech detected, stop collecting
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield b''.join(voiced_frames)
                    ring_buffer.clear()
                    voiced_frames = []
                    
        # Return any remaining voiced audio frames
        if voiced_frames:
            yield b''.join(voiced_frames)
    
    def process_audio(self, audio_path: str, output_path: str = None) -> str:
        """
        Process audio to detect voice activity and filter non-speech segments.
        
        Args:
            audio_path (str): Path to input audio file
            output_path (str): Path to save processed audio
            
        Returns:
            str: Path to processed audio file
        """
        start_time = time.time()
        
        # Ensure the model is loaded
        if self.model is None:
            self.load_model()
        
        # Create output path if not provided
        if output_path is None:
            base_dir = os.path.dirname(audio_path)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(base_dir, f"{base_name}_vad.wav")
        
        try:
            # Load audio file
            logger.info(f"Loading audio file: {audio_path}")
            audio, orig_sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Resample to 16kHz if needed (WebRTC VAD supports 8kHz, 16kHz, 32kHz, 48kHz)
            if orig_sr != self.sample_rate:
                logger.info(f"Resampling from {orig_sr}Hz to {self.sample_rate}Hz")
                audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=self.sample_rate)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio)) * 0.9
            
            # Generate audio frames
            logger.info("Generating audio frames for VAD processing")
            frames = list(self.frame_generator(audio, self.frame_duration_ms, self.sample_rate))
            
            # Apply VAD
            logger.info("Applying voice activity detection")
            voiced_segments = list(self.vad_collector(
                self.sample_rate, 
                self.frame_duration_ms, 
                self.frame_duration_ms * 10,  # 300ms padding
                frames
            ))
            
            if not voiced_segments:
                logger.warning("No voice segments detected in the audio")
                # If no voice segments found, return original audio
                sf.write(output_path, audio, self.sample_rate)
                self.metrics["processing_time"] = time.time() - start_time
                return output_path
            
            # Convert processed segments back to numpy array
            logger.info(f"Found {len(voiced_segments)} voiced segments")
            processed_audio = np.array([])
            
            for segment in voiced_segments:
                # Convert segment bytes to int16 values
                segment_array = np.frombuffer(segment, dtype=np.int16)
                # Convert to float and normalize
                segment_float = segment_array.astype(np.float32) / 32767.0
                # Append to processed audio
                processed_audio = np.append(processed_audio, segment_float)
            
            # Save processed audio
            logger.info(f"Saving processed audio to: {output_path}")
            sf.write(output_path, processed_audio, self.sample_rate)
            
            # Update metrics
            self.metrics["processing_time"] = time.time() - start_time
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error in VAD processing: {e}")
            self.metrics["processing_time"] = time.time() - start_time
            raise