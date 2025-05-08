from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class VoiceProfileBase(BaseModel):
    name: str
    category: Optional[str] = None

class VoiceProfileCreate(VoiceProfileBase):
    pass

class VoiceProfile(VoiceProfileBase):
    id: int
    created_at: datetime
    file_path: str
    embedding_path: str

    class Config:
        from_attributes = True  # Updated from orm_mode
