"""
Style Profile Builder for MidiDrumiGen v2.0 Research Pipeline

This module aggregates research data from all collectors and generates
comprehensive style profiles with vector embeddings for LLM-based generation.

Part of: Epic 1 - Research Pipeline (E1.S5)
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.database.models import StyleProfile
from src.research.collectors.base import ResearchSource

logger = logging.getLogger(__name__)


class StyleProfileBuilder:
    """
    Aggregates research data from all collectors and generates comprehensive
    style profiles with vector embeddings.

    The builder receives data from four collector types (papers, articles,
    audio analysis, MIDI databases) and synthesizes a complete StyleProfile
    with text descriptions, quantitative parameters, and embeddings.
    """

    # Confidence weights by source type
    SOURCE_WEIGHTS = {
        "paper": 0.30,
        "article": 0.20,
        "audio": 0.35,
        "midi": 0.15,
    }

    # Embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, timeout: int = 60):
        """
        Initialize StyleProfileBuilder.

        Args:
            timeout: Timeout in seconds for embedding generation
        """
        self.timeout = timeout
        self._model = None  # Lazy-loaded

    def _get_embedding_model(self) -> SentenceTransformer:
        """
        Lazy-load the SentenceTransformer model.

        Returns:
            SentenceTransformer model instance
        """
        if self._model is None:
            try:
                logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
                self._model = SentenceTransformer(self.EMBEDDING_MODEL)
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._model

    async def build(
        self,
        artist_name: str,
        sources: list[list[ResearchSource]],
        artist_id: uuid.UUID | None = None,
    ) -> StyleProfile:
        """
        Build a complete StyleProfile from aggregated research sources.

        Args:
            artist_name: Name of the artist
            sources: List of lists of ResearchSource, one list per collector
            artist_id: Optional artist UUID (generated if not provided)

        Returns:
            Complete StyleProfile object ready for database persistence
        """
        logger.info(f"Building style profile for: {artist_name}")

        # Flatten and categorize sources
        all_sources = self._flatten_sources(sources)
        categorized = self._categorize_sources(all_sources)

        total_sources = len(all_sources)
        logger.info(
            f"Processing {total_sources} sources: "
            f"{len(categorized.get('paper', []))} papers, "
            f"{len(categorized.get('article', []))} articles, "
            f"{len(categorized.get('audio', []))} audio, "
            f"{len(categorized.get('midi', []))} midi"
        )

        # Extract quantitative parameters
        quantitative_params = self._extract_quantitative_params(categorized)

        # Generate text description
        text_description = self._generate_text_description(
            artist_name, categorized, quantitative_params
        )

        # Generate embedding
        embedding = self._generate_embedding(text_description)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(categorized)

        # Extract MIDI template paths
        midi_templates = self._extract_midi_templates(categorized.get("midi", []))

        # Build sources count
        sources_count = {
            "papers": len(categorized.get("paper", [])),
            "articles": len(categorized.get("article", [])),
            "audio": len(categorized.get("audio", [])),
            "midi": len(categorized.get("midi", [])),
        }

        # Create StyleProfile object
        profile = StyleProfile(
            id=uuid.uuid4(),
            artist_id=artist_id or uuid.uuid4(),
            text_description=text_description,
            quantitative_params=quantitative_params,
            midi_templates_json=midi_templates,
            embedding=embedding.tolist() if embedding is not None else None,
            confidence_score=confidence_score,
            sources_count=sources_count,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

        if confidence_score < 0.6:
            logger.warning(
                f"Low confidence score ({confidence_score:.2f}) for {artist_name}. "
                "Consider augmenting with more sources."
            )

        logger.info(f"Built style profile for {artist_name} with confidence {confidence_score:.2f}")
        return profile

    def _flatten_sources(self, sources: list[list[ResearchSource]]) -> list[ResearchSource]:
        """Flatten nested list of sources into single list."""
        flattened = []
        for source_list in sources:
            if source_list:
                flattened.extend(source_list)
        return flattened

    def _categorize_sources(self, sources: list[ResearchSource]) -> dict[str, list[ResearchSource]]:
        """
        Categorize sources by type.

        Args:
            sources: List of all research sources

        Returns:
            Dictionary mapping source_type to list of sources
        """
        categorized: dict[str, list[ResearchSource]] = {}
        for source in sources:
            source_type = source.source_type
            if source_type not in categorized:
                categorized[source_type] = []
            categorized[source_type].append(source)
        return categorized

    def _extract_quantitative_params(
        self, categorized: dict[str, list[ResearchSource]]
    ) -> dict[str, Any]:
        """
        Extract quantitative parameters from all sources.

        Args:
            categorized: Sources categorized by type

        Returns:
            Dictionary of quantitative parameters
        """
        # Extract tempo values
        tempos = self._extract_tempo_values(categorized)
        tempo_params = self._aggregate_tempos(tempos)

        # Extract swing percentage (primarily from audio)
        swing_values = []
        for source in categorized.get("audio", []):
            if "swing" in source.extracted_data:
                swing_values.append(source.extracted_data["swing"])
            elif "swing_percent" in source.extracted_data:
                swing_values.append(source.extracted_data["swing_percent"])
            elif "swing_ratio" in source.extracted_data:
                # Convert ratio to percentage
                ratio = source.extracted_data["swing_ratio"]
                swing_values.append((ratio - 0.5) * 200)  # 0.5 = 0%, 0.67 = 34%

        swing_percent = np.mean(swing_values) if swing_values else 0.0

        # Extract ghost note probability (primarily from MIDI)
        ghost_probs = []
        for source in categorized.get("midi", []):
            if "ghost_note_prob" in source.extracted_data:
                ghost_probs.append(source.extracted_data["ghost_note_prob"])
            elif "ghost_notes" in source.extracted_data:
                ghost_probs.append(source.extracted_data["ghost_notes"])

        ghost_note_prob = np.mean(ghost_probs) if ghost_probs else 0.1

        # Extract syncopation level (from audio analysis)
        syncopation_values = []
        for source in categorized.get("audio", []):
            if "syncopation" in source.extracted_data:
                syncopation_values.append(source.extracted_data["syncopation"])
            elif "syncopation_level" in source.extracted_data:
                syncopation_values.append(source.extracted_data["syncopation_level"])

        syncopation_level = np.mean(syncopation_values) if syncopation_values else 0.5

        # Extract velocity statistics
        velocity_values = []
        for source_type in ["midi", "audio"]:
            for source in categorized.get(source_type, []):
                if "velocity_mean" in source.extracted_data:
                    velocity_values.append(source.extracted_data["velocity_mean"])
                elif "velocities" in source.extracted_data:
                    vels = source.extracted_data["velocities"]
                    if vels:
                        velocity_values.extend(vels if isinstance(vels, list) else [vels])

        velocity_mean = int(np.mean(velocity_values)) if velocity_values else 90
        velocity_std = int(np.std(velocity_values)) if len(velocity_values) > 1 else 15

        return {
            **tempo_params,
            "swing_percent": round(float(swing_percent), 2),
            "ghost_note_prob": round(float(ghost_note_prob), 3),
            "syncopation_level": round(float(syncopation_level), 3),
            "velocity_mean": velocity_mean,
            "velocity_std": velocity_std,
        }

    def _extract_tempo_values(self, categorized: dict[str, list[ResearchSource]]) -> list[float]:
        """
        Extract all tempo values from sources.

        Args:
            categorized: Sources categorized by type

        Returns:
            List of tempo values
        """
        tempos = []

        # From papers - look for BPM mentions in extracted_data
        for source in categorized.get("paper", []):
            if "tempo" in source.extracted_data:
                tempo = source.extracted_data["tempo"]
                if isinstance(tempo, int | float):
                    tempos.append(float(tempo))
                elif isinstance(tempo, list):
                    tempos.extend([float(t) for t in tempo])

        # From articles
        for source in categorized.get("article", []):
            if "tempo" in source.extracted_data:
                tempo = source.extracted_data["tempo"]
                if isinstance(tempo, int | float):
                    tempos.append(float(tempo))
                elif isinstance(tempo, list):
                    tempos.extend([float(t) for t in tempo])

        # From audio analysis
        for source in categorized.get("audio", []):
            if "tempo" in source.extracted_data:
                tempos.append(float(source.extracted_data["tempo"]))
            elif "bpm" in source.extracted_data:
                tempos.append(float(source.extracted_data["bpm"]))

        # From MIDI
        for source in categorized.get("midi", []):
            if "tempo" in source.extracted_data:
                tempos.append(float(source.extracted_data["tempo"]))
            elif "bpm" in source.extracted_data:
                tempos.append(float(source.extracted_data["bpm"]))

        return tempos

    def _aggregate_tempos(self, tempos: list[float]) -> dict[str, int]:
        """
        Aggregate tempo values with outlier filtering.

        Filters outliers beyond 2 standard deviations from the mean.

        Args:
            tempos: List of tempo values

        Returns:
            Dictionary with tempo_min, tempo_max, tempo_avg
        """
        if not tempos:
            return {
                "tempo_min": 90,
                "tempo_max": 120,
                "tempo_avg": 105,
            }

        if len(tempos) == 1:
            tempo = int(tempos[0])
            return {
                "tempo_min": tempo,
                "tempo_max": tempo,
                "tempo_avg": tempo,
            }

        # Filter outliers beyond 2 std dev
        tempos_array = np.array(tempos)
        mean = np.mean(tempos_array)
        std = np.std(tempos_array)

        if std > 0:
            # Keep values within 2 standard deviations
            filtered = tempos_array[np.abs(tempos_array - mean) <= 2 * std]
            if len(filtered) == 0:
                # If all filtered out, use original
                filtered = tempos_array
        else:
            # No variance, use all values
            filtered = tempos_array

        return {
            "tempo_min": int(np.min(filtered)),
            "tempo_max": int(np.max(filtered)),
            "tempo_avg": int(np.mean(filtered)),
        }

    def _generate_text_description(
        self,
        artist_name: str,
        categorized: dict[str, list[ResearchSource]],
        params: dict[str, Any],
    ) -> str:
        """
        Generate a text description suitable for LLM prompts.

        Args:
            artist_name: Name of the artist
            categorized: Sources categorized by type
            params: Quantitative parameters

        Returns:
            Text description (100-300 words)
        """
        # Collect key characteristics from papers
        paper_insights = []
        for source in categorized.get("paper", []):
            if source.extracted_data.get("summary"):
                paper_insights.append(source.extracted_data["summary"])
            if source.extracted_data.get("characteristics"):
                chars = source.extracted_data["characteristics"]
                if isinstance(chars, list):
                    paper_insights.extend(chars)
                else:
                    paper_insights.append(str(chars))

        # Collect technique descriptions from articles
        technique_descriptions = []
        equipment_mentions = []
        style_keywords = []

        for source in categorized.get("article", []):
            if source.extracted_data.get("techniques"):
                techs = source.extracted_data["techniques"]
                if isinstance(techs, list):
                    technique_descriptions.extend(techs)
                else:
                    technique_descriptions.append(str(techs))

            if source.extracted_data.get("equipment"):
                equip = source.extracted_data["equipment"]
                if isinstance(equip, list):
                    equipment_mentions.extend(equip)
                else:
                    equipment_mentions.append(str(equip))

            if source.extracted_data.get("style_keywords"):
                kw = source.extracted_data["style_keywords"]
                if isinstance(kw, list):
                    style_keywords.extend(kw)
                else:
                    style_keywords.append(str(kw))

        # Build description
        description_parts = [f"{artist_name} drumming style profile:"]

        # Tempo info
        tempo_desc = f"Typical tempo range {params['tempo_min']}-{params['tempo_max']} BPM (average {params['tempo_avg']} BPM)."
        description_parts.append(tempo_desc)

        # Swing and feel
        if params["swing_percent"] > 5:
            swing_desc = f"Notable swing feel at {params['swing_percent']:.1f}%."
        else:
            swing_desc = "Relatively straight timing with minimal swing."
        description_parts.append(swing_desc)

        # Ghost notes
        if params["ghost_note_prob"] > 0.2:
            ghost_desc = "Frequent use of ghost notes adding texture and groove."
        elif params["ghost_note_prob"] > 0.1:
            ghost_desc = "Moderate use of ghost notes."
        else:
            ghost_desc = "Sparse ghost note usage."
        description_parts.append(ghost_desc)

        # Syncopation
        if params["syncopation_level"] > 0.7:
            sync_desc = "Highly syncopated patterns with complex rhythmic displacement."
        elif params["syncopation_level"] > 0.4:
            sync_desc = "Moderate syncopation creating rhythmic interest."
        else:
            sync_desc = "Straightforward rhythmic patterns with minimal syncopation."
        description_parts.append(sync_desc)

        # Velocity dynamics
        if params["velocity_std"] > 20:
            vel_desc = f"Wide dynamic range with expressive velocity variations (mean {params['velocity_mean']})."
        else:
            vel_desc = f"Consistent velocity around {params['velocity_mean']}."
        description_parts.append(vel_desc)

        # Add paper insights (limit to 2)
        if paper_insights:
            description_parts.append("Research findings: " + " ".join(paper_insights[:2]))

        # Add techniques (limit to 3)
        if technique_descriptions:
            unique_techs = list(set(technique_descriptions))[:3]
            description_parts.append("Key techniques: " + ", ".join(unique_techs) + ".")

        # Add equipment (limit to 3)
        if equipment_mentions:
            unique_equip = list(set(equipment_mentions))[:3]
            description_parts.append("Equipment: " + ", ".join(unique_equip) + ".")

        # Add style keywords (limit to 5)
        if style_keywords:
            unique_kw = list(set(style_keywords))[:5]
            description_parts.append("Style: " + ", ".join(unique_kw) + ".")

        return " ".join(description_parts)

    def _generate_embedding(self, text: str) -> np.ndarray | None:
        """
        Generate vector embedding for the text description.

        Args:
            text: Text description to embed

        Returns:
            384-dimensional numpy array or None if failed
        """
        try:
            model = self._get_embedding_model()
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def _calculate_confidence_score(self, categorized: dict[str, list[ResearchSource]]) -> float:
        """
        Calculate weighted confidence score based on source types and counts.

        Args:
            categorized: Sources categorized by type

        Returns:
            Confidence score between 0.0 and 1.0
        """
        total_score = 0.0

        for source_type, weight in self.SOURCE_WEIGHTS.items():
            sources = categorized.get(source_type, [])
            if not sources:
                continue

            # Calculate average confidence of sources
            avg_confidence = np.mean([s.confidence for s in sources])

            # Apply diminishing returns for excessive sources
            # Optimal: 3-5 sources per type
            source_count = len(sources)
            if source_count <= 5:
                count_factor = source_count / 5.0
            else:
                # Diminishing returns beyond 5
                count_factor = 1.0 + (np.log10(source_count / 5.0) * 0.2)
                count_factor = min(count_factor, 1.3)  # Cap at 1.3

            # Combine confidence and count
            type_score = avg_confidence * count_factor * weight
            total_score += type_score

        # Normalize and clamp
        confidence = min(max(total_score, 0.0), 1.0)

        return round(confidence, 3)

    def _extract_midi_templates(self, midi_sources: list[ResearchSource]) -> list[str]:
        """
        Extract MIDI file paths from MIDI sources.

        Args:
            midi_sources: List of MIDI research sources

        Returns:
            List of file paths to MIDI templates
        """
        templates = []
        for source in midi_sources:
            if source.file_path:
                templates.append(source.file_path)
        return templates
