from datetime import datetime
import uuid

from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class BaseModel(Base):
    __tablename__ = 'base_models'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, unique=True)
    hf_model_id = Column(String(255), nullable=True)  # e.g. "Qwen/Qwen2.5-0.5B-Instruct"
    model_download_link = Column(String(500), nullable=True)  # GGUF base model download URL (user fills after upload)
    local_model_path = Column(String(500), nullable=True)  # On-device path (optional)
    version = Column(String(50), nullable=False)
    size_mb = Column(Integer)
    is_active = Column(Boolean, default=True, server_default='true')
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now())
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, server_default=func.now())

    adapters = relationship("Adapter", back_populates="base_model", cascade="all, delete-orphan")


class Adapter(Base):
    __tablename__ = 'adapters'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    base_model_id = Column(String, ForeignKey('base_models.id', ondelete='CASCADE'), nullable=False)
    name = Column(String(255), nullable=False)
    domain = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    storage_path = Column(Text, nullable=False)
    size_mb = Column(Float)
    checksum_sha256 = Column(String(64))
    training_epochs = Column(Integer)
    training_loss = Column(Float)
    status = Column(String(50), default='pending', server_default='pending')
    is_published = Column(Boolean, default=False, server_default='false')
    created_at = Column(DateTime, default=datetime.utcnow, server_default=func.now())
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, server_default=func.now())

    base_model = relationship("BaseModel", back_populates="adapters")
    deployments = relationship("AdapterDeployment", back_populates="adapter")


class AdapterDeployment(Base):
    __tablename__ = 'adapter_deployments'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    adapter_id = Column(String, ForeignKey('adapters.id', ondelete='CASCADE'), nullable=False)
    rollout_percentage = Column(Integer, default=100, server_default='100')
    min_app_version = Column(String(50))
    target_devices = Column(Text)
    is_active = Column(Boolean, default=True, server_default='true')
    deployed_at = Column(DateTime, default=datetime.utcnow, server_default=func.now())

    adapter = relationship("Adapter", back_populates="deployments")


class UpdateLog(Base):
    __tablename__ = 'update_logs'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    device_id = Column(String(255), nullable=False)
    adapter_id = Column(String, ForeignKey('adapters.id'), nullable=False)
    downloaded_at = Column(DateTime, default=datetime.utcnow, server_default=func.now())
    installation_status = Column(String(50), default='pending', server_default='pending')
    error_message = Column(Text)
