"""
Database models for MidiDrumiGen v2.0.

SQLAlchemy models with pgvector support for artist style profiles,
research sources, and generation history.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Artist(Base):
    """
    Artist/band/producer entity with research status tracking.

    Central entity linking to style profiles and research sources.
    """

    __tablename__ = "artists"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, index=True, nullable=False)
    research_status = Column(
        String(50), default="pending", nullable=False, comment="pending/researching/cached/failed"
    )
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sources_count = Column(Integer, default=0)
    confidence_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    style_profile = relationship("StyleProfile", back_populates="artist", uselist=False)
    research_sources = relationship("ResearchSource", back_populates="artist")
    generation_history = relationship("GenerationHistory", back_populates="artist")

    def __repr__(self):
        return f"<Artist(name='{self.name}', status='{self.research_status}')>"


class ResearchSource(Base):
    """
    Individual research source (paper, article, audio, MIDI) for an artist.

    Stores raw content and extracted data from each collector.
    """

    __tablename__ = "research_sources"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artist_id = Column(UUID(as_uuid=True), ForeignKey("artists.id"), nullable=False)
    source_type = Column(String(50), nullable=False, comment="paper/article/audio/midi")
    url = Column(String, nullable=True)
    file_path = Column(String, nullable=True)
    raw_content = Column(String, nullable=True)
    extracted_data = Column(JSON, nullable=True)
    confidence = Column(Float, default=0.5)
    collected_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    artist = relationship("Artist", back_populates="research_sources")

    # Indexes
    __table_args__ = (Index("idx_research_sources_artist_type", "artist_id", "source_type"),)

    def __repr__(self):
        return f"<ResearchSource(type='{self.source_type}', artist_id='{self.artist_id}')>"


class StyleProfile(Base):
    """
    Aggregated style profile for an artist with vector embeddings.

    Contains textual description, quantitative parameters, MIDI templates,
    and sentence-transformers embedding for similarity search.
    """

    __tablename__ = "style_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artist_id = Column(UUID(as_uuid=True), ForeignKey("artists.id"), unique=True, nullable=False)

    # Text description for LLM prompts
    text_description = Column(String, nullable=False)

    # Quantitative parameters (JSON)
    # Format: {tempo_min, tempo_max, tempo_avg, swing_percent, ghost_note_prob, etc.}
    quantitative_params = Column(JSON, nullable=False)

    # MIDI template file paths (JSON array)
    midi_templates_json = Column(JSON, default=list)

    # Vector embedding for similarity search (384 dimensions for all-MiniLM-L6-v2)
    embedding = Column(Vector(384), nullable=True)

    # Quality metrics
    confidence_score = Column(Float, nullable=False)
    sources_count = Column(JSON, default=dict)  # {papers: 5, articles: 10, audio: 3, midi: 2}

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    artist = relationship("Artist", back_populates="style_profile")

    # Indexes for vector similarity search
    __table_args__ = (
        Index(
            "idx_style_profiles_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self):
        return (
            f"<StyleProfile(artist_id='{self.artist_id}', confidence={self.confidence_score:.2f})>"
        )


class GenerationHistory(Base):
    """
    History of MIDI generation requests for analytics and monitoring.

    Tracks which LLM provider was used, generation time, cost, and output files.
    """

    __tablename__ = "generation_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    artist_id = Column(UUID(as_uuid=True), ForeignKey("artists.id"), nullable=False)

    # Provider info
    provider_used = Column(String(50), nullable=False, comment="anthropic/google/openai/template")

    # Performance metrics
    generation_time_ms = Column(Integer, nullable=False)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)

    # Request parameters
    user_params = Column(JSON, nullable=False)  # {bars, tempo, time_signature, variations}

    # Output files (JSON array of paths)
    output_files = Column(JSON, default=list)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    artist = relationship("Artist", back_populates="generation_history")

    # Indexes
    __table_args__ = (
        Index("idx_generation_history_artist", "artist_id"),
        Index("idx_generation_history_created", "created_at"),
    )

    def __repr__(self):
        return (
            f"<GenerationHistory(artist_id='{self.artist_id}', "
            f"provider='{self.provider_used}', time={self.generation_time_ms}ms)>"
        )
