"""Initial v2.0 schema

Revision ID: 6ad616b97364
Revises:
Create Date: 2025-11-17 20:24:26.802881

"""
from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '6ad616b97364'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Create artists table
    op.create_table(
        'artists',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('name', sa.String(255), unique=True, nullable=False),
        sa.Column('research_status', sa.String(50), server_default='pending'),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('sources_count', sa.Integer(), server_default='0'),
        sa.Column('confidence_score', sa.Float(), server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_artist_name', 'artists', ['name'])
    op.create_index('idx_research_status', 'artists', ['research_status'])

    # Create research_sources table
    op.create_table(
        'research_sources',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('artist_id', sa.Integer(), sa.ForeignKey('artists.id', ondelete='CASCADE'), nullable=False),
        sa.Column('source_type', sa.String(50), nullable=False),
        sa.Column('url', sa.Text(), nullable=True),
        sa.Column('file_path', sa.Text(), nullable=True),
        sa.Column('raw_content', sa.Text(), nullable=True),
        sa.Column('extracted_data', sa.JSON(), nullable=True),
        sa.Column('confidence', sa.Float(), server_default='0.5'),
        sa.Column('collected_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_artist_id', 'research_sources', ['artist_id'])
    op.create_index('idx_source_type', 'research_sources', ['source_type'])

    # Create style_profiles table
    op.create_table(
        'style_profiles',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('artist_id', sa.Integer(), sa.ForeignKey('artists.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('text_description', sa.Text(), nullable=False),
        sa.Column('quantitative_params', sa.JSON(), nullable=False),
        sa.Column('midi_templates_json', sa.JSON(), nullable=True),
        sa.Column('embedding', Vector(384), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now())
    )
    op.create_index('idx_profile_artist_id', 'style_profiles', ['artist_id'])

    # Create vector index for similarity search
    op.execute('CREATE INDEX ON style_profiles USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)')

    # Create generation_history table
    op.create_table(
        'generation_history',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('artist_id', sa.Integer(), sa.ForeignKey('artists.id', ondelete='SET NULL'), nullable=True),
        sa.Column('provider_used', sa.String(50), nullable=True),
        sa.Column('generation_time_ms', sa.Integer(), nullable=True),
        sa.Column('user_params', sa.JSON(), nullable=True),
        sa.Column('output_files', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_gen_artist_id', 'generation_history', ['artist_id'])
    op.create_index('idx_gen_created_at', 'generation_history', ['created_at'])

    # Create user_sessions table (future)
    op.create_table(
        'user_sessions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('session_id', sa.String(255), unique=True, nullable=False),
        sa.Column('user_ip', sa.String(45), nullable=True),
        sa.Column('requests_count', sa.Integer(), server_default='0'),
        sa.Column('last_request', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now())
    )
    op.create_index('idx_session_id', 'user_sessions', ['session_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_session_id', 'user_sessions')
    op.drop_table('user_sessions')

    op.drop_index('idx_gen_created_at', 'generation_history')
    op.drop_index('idx_gen_artist_id', 'generation_history')
    op.drop_table('generation_history')

    op.execute('DROP INDEX IF EXISTS style_profiles_embedding_idx')
    op.drop_index('idx_profile_artist_id', 'style_profiles')
    op.drop_table('style_profiles')

    op.drop_index('idx_source_type', 'research_sources')
    op.drop_index('idx_artist_id', 'research_sources')
    op.drop_table('research_sources')

    op.drop_index('idx_research_status', 'artists')
    op.drop_index('idx_artist_name', 'artists')
    op.drop_table('artists')

    op.execute('DROP EXTENSION IF EXISTS vector')
