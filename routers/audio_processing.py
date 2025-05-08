# Updated routers/audio_processing.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import os
import time
import logging
from database import SessionLocal
from utils.audio_processing import isolate_voice
from utils.audio_conversion import convert_audio_to_wav, is_valid_wav

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audio", tags=["audio"])

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Directory setup
TEMP_DIR = "data/temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/process")
async def process_audio(
    profile_id: int = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Process audio file to isolate a voice based on profile."""
    logger.info(f"Received audio processing request for profile ID: {profile_id}")
    logger.info(f"Uploaded file: {audio_file.filename}, content type: {audio_file.content_type}")
    
    # Create temporary file for uploaded audio - preserve original extension
    timestamp = int(time.time())
    file_ext = os.path.splitext(audio_file.filename)[1]
    
    # If no extension provided in filename, try to determine from content type
    if not file_ext:
        if "webm" in audio_file.content_type:
            file_ext = ".webm"
        elif "wav" in audio_file.content_type:
            file_ext = ".wav"
        elif "mp3" in audio_file.content_type:
            file_ext = ".mp3"
        elif "ogg" in audio_file.content_type or "opus" in audio_file.content_type:
            file_ext = ".ogg"
        else:
            file_ext = ".audio"  # Generic extension
    
    temp_orig_path = os.path.join(TEMP_DIR, f"temp_orig_{timestamp}{file_ext}")
    temp_wav_path = os.path.join(TEMP_DIR, f"temp_{timestamp}.wav")
    
    try:
        # Save uploaded file with original format
        logger.info(f"Saving uploaded file to: {temp_orig_path}")
        content = await audio_file.read()
        with open(temp_orig_path, "wb") as buffer:
            buffer.write(content)
        
        # Verify file was saved
        if not os.path.exists(temp_orig_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
            
        file_size = os.path.getsize(temp_orig_path)
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Check if it's already a valid WAV
        is_wav = file_ext.lower() == ".wav" and is_valid_wav(temp_orig_path)
        
        if is_wav:
            # If it's already a valid WAV file, just use the original
            logger.info("File is already a valid WAV - skipping conversion")
            temp_wav_path = temp_orig_path
        else:
            # Convert to WAV format
            logger.info(f"Converting {file_ext} to WAV format")
            try:
                temp_wav_path = convert_audio_to_wav(
                    temp_orig_path, 
                    temp_wav_path,
                    sample_rate=16000  # 16kHz sample rate for voice processing
                )
            except Exception as e:
                logger.error(f"Audio conversion error: {str(e)}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Could not convert audio file: {str(e)}"
                )
        
        # Process the audio file to isolate the voice
        logger.info("Starting voice isolation")
        output_dir = "data/isolated"
        
        try:
            # Important: Pass parameters in the correct order
            isolated_path, similarity = isolate_voice(
                temp_wav_path,
                profile_id,
                db,  # This parameter will be passed as db_session
                output_dir=output_dir,
                use_gpu=False
            )
            
            # Get the filename for response
            isolated_filename = os.path.basename(isolated_path)
            logger.info(f"Voice isolation complete. Result file: {isolated_filename}")
            
            # Return response
            return {
                "status": "success",
                "message": "Audio processed successfully",
                "file_name": isolated_filename,
                "similarity": float(similarity),
                "file_path": f"/data/isolated/{isolated_filename}"
            }
        except Exception as e:
            logger.error(f"Voice isolation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Voice isolation failed: {str(e)}"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full error for other exceptions
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        
        # Return a helpful error message
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up temporary files
        for path in [temp_orig_path, temp_wav_path]:
            if os.path.exists(path) and os.path.basename(path).startswith("temp_"):
                try:
                    os.remove(path)
                    logger.info(f"Cleaned up temporary file: {path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {e}")