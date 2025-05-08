from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from database import Base

class VoiceProfile(Base):
    __tablename__ = "voice_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True, nullable=False)
    category = Column(String(50), nullable=True)
    file_path = Column(String(255), nullable=False)
    embedding_path = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    def __repr__(self):
        return f"<VoiceProfile(id={self.id}, name='{self.name}', category='{self.category}')>"
