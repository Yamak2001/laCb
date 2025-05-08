from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
import os
import shutil
import time
import logging
from database import SessionLocal
from utils.audio_processing import isolate_voice

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
    
    # Create temporary file for uploaded audio
    timestamp = int(time.time())
    temp_audio_path = os.path.join(TEMP_DIR, f"temp_{timestamp}.wav")
    
    try:
        # Save uploaded file
        logger.info(f"Saving uploaded file to: {temp_audio_path}")
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Verify file was saved
        if not os.path.exists(temp_audio_path):
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
            
        file_size = os.path.getsize(temp_audio_path)
        logger.info(f"File saved successfully. Size: {file_size} bytes")
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        # Process the audio file to isolate the voice
        logger.info("Starting voice isolation")
        output_dir = "data/isolated"
        isolated_path, similarity = isolate_voice(
            temp_audio_path,
            profile_id,
            db,
            output_dir=output_dir,
            use_gpu=False  # Set to True if GPU is available
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
        # Log the full error
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        
        # Clean up on error
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
        # Return a helpful error message
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Cleaned up temporary file: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")