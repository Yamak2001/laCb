from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import numpy as np
import shutil
import time
from resemblyzer import VoiceEncoder, preprocess_wav

from database import SessionLocal
from models.voice_profile import VoiceProfile
from schemas.voice_profile import VoiceProfile as VoiceProfileSchema
from schemas.voice_profile import VoiceProfileCreate

router = APIRouter(prefix="/api/voice-profiles", tags=["voice-profiles"])

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Directory setup
AUDIO_DIR = "data/audio"
EMBEDDING_DIR = "data/embeddings"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

@router.get("/", response_model=List[VoiceProfileSchema])
def get_voice_profiles(db: Session = Depends(get_db)):
    profiles = db.query(VoiceProfile).all()
    return profiles

@router.get("/{profile_id}", response_model=VoiceProfileSchema)
def get_voice_profile(profile_id: int, db: Session = Depends(get_db)):
    profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if profile is None:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    return profile

@router.post("/", response_model=VoiceProfileSchema)
async def create_voice_profile(
    name: str = Form(...),
    category: Optional[str] = Form(None),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save audio file
    timestamp = int(time.time())
    safe_name = name.replace(" ", "_").lower()
    audio_filename = f"{safe_name}_{timestamp}.wav"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)
    
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    try:
        # Process audio to extract voice embedding
        wav = preprocess_wav(audio_path)
        encoder = VoiceEncoder()
        embedding = encoder.embed_utterance(wav)
        
        # Save embedding
        embedding_filename = f"{safe_name}_{timestamp}.npy"
        embedding_path = os.path.join(EMBEDDING_DIR, embedding_filename)
        np.save(embedding_path, embedding)
        
        # Create profile in database
        db_profile = VoiceProfile(
            name=name,
            category=category,
            file_path=audio_path,
            embedding_path=embedding_path
        )
        db.add(db_profile)
        db.commit()
        db.refresh(db_profile)
        return db_profile
        
    except Exception as e:
        # Clean up if processing fails
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")

@router.put("/{profile_id}", response_model=VoiceProfileSchema)
def update_voice_profile(
    profile_id: int, 
    profile_update: VoiceProfileCreate, 
    db: Session = Depends(get_db)
):
    db_profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if db_profile is None:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    
    # Update fields
    db_profile.name = profile_update.name
    db_profile.category = profile_update.category
    
    db.commit()
    db.refresh(db_profile)
    return db_profile

@router.delete("/{profile_id}")
def delete_voice_profile(profile_id: int, db: Session = Depends(get_db)):
    db_profile = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).first()
    if db_profile is None:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    
    # Delete files
    if os.path.exists(db_profile.file_path):
        os.remove(db_profile.file_path)
    if os.path.exists(db_profile.embedding_path):
        os.remove(db_profile.embedding_path)
    
    db.delete(db_profile)
    db.commit()
    return {"message": "Profile deleted successfully"}
