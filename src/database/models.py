"""
Database models for document tracking system.
"""
from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Document(Base):
    """Represents a tracked document."""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    classification = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    current_version_id = Column(Integer, ForeignKey("versions.id"))
    parent_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    
    versions = relationship("Version", back_populates="document")
    current_version = relationship("Version", foreign_keys=[current_version_id])
    children = relationship("Document", backref=backref("parent", remote_side=[id]))
    related_documents = relationship("DocumentRelationship", back_populates="document")
    
    def __repr__(self):
        return f"Document(id={self.id}, path={self.path}, title={self.title})"

class DocumentRelationship(Base):
    """Represents relationships between documents."""
    __tablename__ = "document_relationships"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    related_document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    relationship_type = Column(String, nullable=False)  # e.g., 'related', 'supplements', 'references'
    description = Column(Text, nullable=True)
    
    document = relationship("Document", foreign_keys=[document_id], back_populates="related_documents")
    related_document = relationship("Document", foreign_keys=[related_document_id])
    
    def __repr__(self):
        return f"DocumentRelationship(id={self.id}, type={self.relationship_type})"

class Version(Base):
    """Represents a version of a document."""
    __tablename__ = "versions"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String, nullable=False)
    metadata = Column(Text, nullable=False)  # JSON string
    changes = Column(Text, nullable=True)    # JSON string
    reverted_from = Column(Integer, ForeignKey("versions.id"), nullable=True)
    
    document = relationship("Document", back_populates="versions")
    
    def __repr__(self):
        return f"Version(id={self.id}, document_id={self.document_id}, version={self.version_number})"

class Notification(Base):
    """Represents a notification event."""
    __tablename__ = "notifications"
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True)
    version_id = Column(Integer, ForeignKey("versions.id"), nullable=True)
    
    document = relationship("Document")
    version = relationship("Version")
    
    def __repr__(self):
        return f"Notification(id={self.id}, type={self.event_type}, message={self.message[:50]})"
