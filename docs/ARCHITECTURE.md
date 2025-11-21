# System Architecture
# MidiDrumiGen v2.0 - Technical Architecture Document

**Version:** 2.0.0
**Date:** 2025-11-17
**Status:** Design Phase
**Authors:** Engineering Team

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [Data Flow](#data-flow)
4. [Technology Stack](#technology-stack)
5. [Database Schema](#database-schema)
6. [API Specification](#api-specification)
7. [Deployment Architecture](#deployment-architecture)
8. [Security](#security)
9. [Performance Optimization](#performance-optimization)
10. [Monitoring & Observability](#monitoring--observability)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Layer                                │
│  ┌──────────────────┐         ┌──────────────────────┐         │
│  │ Max for Live     │         │ Web Dashboard        │         │
│  │ (Ableton Plugin) │         │ (Future)             │         │
│  └────────┬─────────┘         └──────────┬───────────┘         │
└───────────┼────────────────────────────────┼─────────────────────┘
            │                                │
            │ HTTP/REST                      │ HTTP/REST
            │                                │
┌───────────┴────────────────────────────────┴─────────────────────┐
│                      API Gateway Layer                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              FastAPI REST API Server                     │   │
│  │  - Authentication & Authorization                        │   │
│  │  - Request Validation                                    │   │
│  │  - Rate Limiting                                         │   │
│  │  - Response Caching                                      │   │
│  └──────────────┬───────────────────────────────────────────┘   │
└─────────────────┼───────────────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────────────┐
│                   Orchestrator Layer                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             Main Orchestrator Agent                      │   │
│  │  - Request Routing                                       │   │
│  │  - Cache Management                                      │   │
│  │  - Task Coordination                                     │   │
│  │  - Error Handling & Fallbacks                           │   │
│  └──────┬───────────────────────────────────────┬──────────┘   │
└─────────┼───────────────────────────────────────┼──────────────┘
          │                                       │
          │ Cached?                               │ Not Cached
          │ Yes                                   │
          ▼                                       ▼
┌─────────────────────┐              ┌─────────────────────────────┐
│  Generation Layer   │              │    Research Layer           │
│ ┌─────────────────┐ │              │ ┌─────────────────────────┐ │
│ │ LLM Provider    │ │              │ │ Research Orchestrator   │ │
│ │ Manager         │ │              │ │                         │ │
│ │ - Anthropic     │ │              │ │ ┌─────────┐ ┌─────────┐ │ │
│ │ - Google        │ │              │ │ │ Papers  │ │Articles │ │ │
│ │ - OpenAI        │ │              │ │ │Collector│ │Collector│ │ │
│ │                 │ │              │ │ └─────────┘ └─────────┘ │ │
│ └────────┬────────┘ │              │ │ ┌─────────┐ ┌─────────┐ │ │
│          │          │              │ │ │ Audio   │ │  MIDI   │ │ │
│          ▼          │              │ │ │Analysis │ │Database │ │ │
│ ┌─────────────────┐ │              │ │ └─────────┘ └─────────┘ │ │
│ │ Template-based  │ │              │ └────────────┬────────────┘ │
│ │ Generator       │ │              │              │              │
│ │ (Fallback)      │ │              │              ▼              │
│ └────────┬────────┘ │              │ ┌─────────────────────────┐ │
│          │          │              │ │ Style Profile Builder   │ │
│          ▼          │              │ └───────────┬─────────────┘ │
│ ┌─────────────────┐ │              └─────────────┼───────────────┘
│ │ Hybrid          │ │                            │
│ │ Coordinator     │ │                            │
│ └────────┬────────┘ │                            │
│          │          │                            │
│          ▼          │                            │
│ ┌─────────────────┐ │                            │
│ │ MIDI Export &   │ │                            │
│ │ Validation      │ │                            │
│ └────────┬────────┘ │                            │
└──────────┼──────────┘                            │
           │                                       │
           └───────────────────┬───────────────────┘
                               │
┌──────────────────────────────┴──────────────────────────────────┐
│                      Data Layer                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │  PostgreSQL    │  │     Redis      │  │  File Storage    │  │
│  │  + pgvector    │  │  (Task Queue)  │  │  (MIDI Files)    │  │
│  │                │  │                │  │                  │  │
│  │ - Artists      │  │ - Celery Tasks │  │ - Generated MIDI │  │
│  │ - StyleProfiles│  │ - Rate Limits  │  │ - Templates      │  │
│  │ - Sources      │  │ - Cache        │  │                  │  │
│  └────────────────┘  └────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## System Components

### 1. Orchestrator Agent
**Location:** `src/orchestrator/main.py`

**Responsibilities:**
- Entry point for all user requests
- Cache hit/miss decision making
- Route to Research or Generation pipelines
- Monitor task progress
- Aggregate results
- Handle global errors and fallbacks

**Key Methods:**
```python
class Orchestrator:
    async def process_request(
        self,
        artist_name: str,
        params: GenerationParams
    ) -> GenerationResult:
        """Main entry point for all requests."""

    async def check_cache(self, artist_name: str) -> Optional[StyleProfile]:
        """Check if artist is already researched."""

    async def delegate_research(self, artist_name: str) -> StyleProfile:
        """Delegate to Research Orchestrator."""

    async def delegate_generation(
        self,
        profile: StyleProfile,
        params: GenerationParams
    ) -> List[Path]:
        """Delegate to Generation Coordinator."""
```

---

### 2. Research Orchestrator
**Location:** `src/research/orchestrator.py`

**Responsibilities:**
- Coordinate all data collectors
- Run collectors in parallel (asyncio)
- Aggregate results from all sources
- Call Style Profile Builder
- Store profile in database
- Update progress status

**Architecture:**
```python
class ResearchOrchestrator:
    def __init__(self):
        self.collectors = {
            'papers': ScholarPaperCollector(),
            'articles': WebArticleCollector(),
            'audio': AudioAnalysisCollector(),
            'midi': MidiDatabaseCollector()
        }
        self.profile_builder = StyleProfileBuilder()

    async def research_artist(
        self,
        artist_name: str,
        depth: ResearchDepth = ResearchDepth.FULL
    ) -> StyleProfile:
        # Run all collectors in parallel
        tasks = [
            collector.collect(artist_name)
            for collector in self.collectors.values()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build StyleProfile
        profile = await self.profile_builder.build(artist_name, results)

        # Store in database
        await self.db.save_profile(profile)

        return profile
```

---

### 3. Data Collectors

#### 3.1 Scholar Paper Collector
**Location:** `src/research/collectors/papers.py`

**External APIs:**
- Semantic Scholar API: `https://api.semanticscholar.org/graph/v1/paper/search`
- ArXiv API: `http://export.arxiv.org/api/query`
- CrossRef API: `https://api.crossref.org/works`

**Implementation:**
```python
class ScholarPaperCollector(BaseCollector):
    async def collect(self, artist_name: str) -> List[ResearchSource]:
        papers = []

        # Semantic Scholar
        papers.extend(await self._search_semantic_scholar(artist_name))

        # ArXiv
        papers.extend(await self._search_arxiv(artist_name))

        # CrossRef
        papers.extend(await self._search_crossref(artist_name))

        return papers

    async def _search_semantic_scholar(self, artist: str) -> List[ResearchSource]:
        query = f"{artist} drumming style rhythm analysis"
        url = f"https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            'query': query,
            'fields': 'title,abstract,authors,citationCount,year',
            'limit': 10
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()

        return [
            ResearchSource(
                source_type='paper',
                title=paper['title'],
                url=f"https://www.semanticscholar.org/paper/{paper['paperId']}",
                raw_content=paper['abstract'],
                extracted_data={
                    'authors': paper['authors'],
                    'citations': paper['citationCount'],
                    'year': paper['year']
                },
                confidence=self._calculate_confidence(paper)
            )
            for paper in data.get('data', [])
        ]
```

---

#### 3.2 Web Article Collector
**Location:** `src/research/collectors/articles.py`

**Technologies:**
- BeautifulSoup4 for HTML parsing
- Scrapy for crawling
- Newspaper3k for article extraction
- spaCy for NLP (entity recognition, keyword extraction)

**Target Sites:**
- Pitchfork: `https://pitchfork.com/search/?query={artist}`
- Rolling Stone: `https://www.rollingstone.com/search/articles/{artist}`
- Drummerworld: `https://www.drummerworld.com/drummers/{artist}.html`
- Wikipedia: `https://en.wikipedia.org/wiki/{artist}`

**Implementation:**
```python
class WebArticleCollector(BaseCollector):
    def __init__(self):
        self.sites = [
            PitchforkScraper(),
            RollingStoneScraper(),
            DrummerworldScraper(),
            WikipediaScraper()
        ]

    async def collect(self, artist_name: str) -> List[ResearchSource]:
        articles = []

        for scraper in self.sites:
            try:
                results = await scraper.search(artist_name)
                articles.extend(results)
            except Exception as e:
                logger.warning(f"{scraper.__class__.__name__} failed: {e}")

        # Extract drumming-related content using NLP
        articles = self._filter_drumming_content(articles)

        return articles

    def _filter_drumming_content(self, articles: List) -> List:
        """Use spaCy to find drumming-related content."""
        nlp = spacy.load("en_core_web_sm")
        drumming_keywords = ['drum', 'beat', 'rhythm', 'groove', 'tempo',
                             'kit', 'cymbal', 'snare', 'kick', 'hi-hat']

        filtered = []
        for article in articles:
            doc = nlp(article.raw_content)

            # Count drumming keyword mentions
            keyword_count = sum(
                1 for token in doc if token.text.lower() in drumming_keywords
            )

            if keyword_count >= 3:  # Threshold
                article.confidence *= (1 + keyword_count * 0.05)
                filtered.append(article)

        return filtered
```

---

#### 3.3 Audio Analysis Collector
**Location:** `src/research/collectors/audio.py`

**Technologies:**
- yt-dlp: Download YouTube audio
- Librosa: Tempo/beat detection
- madmom: Advanced beat/tempo tracking with DBN models

**Implementation:**
```python
class AudioAnalysisCollector(BaseCollector):
    async def collect(self, artist_name: str) -> List[ResearchSource]:
        # Search YouTube for performances
        video_urls = await self._search_youtube(artist_name)

        analyses = []
        for url in video_urls[:5]:  # Limit to 5 videos
            try:
                # Download audio
                audio_path = await self._download_audio(url)

                # Analyze
                analysis = await self._analyze_audio(audio_path)

                analyses.append(ResearchSource(
                    source_type='audio',
                    url=url,
                    extracted_data=analysis,
                    confidence=analysis['confidence']
                ))

                # Clean up
                audio_path.unlink()

            except Exception as e:
                logger.warning(f"Audio analysis failed for {url}: {e}")

        return analyses

    async def _analyze_audio(self, audio_path: Path) -> dict:
        """Extract rhythm features from audio using Librosa + madmom."""
        import librosa
        import numpy as np
        from madmom.features.beats import RNNBeatProcessor
        from madmom.features.tempo import TempoEstimationProcessor

        # Load audio
        y, sr = librosa.load(audio_path, sr=44100)

        # Librosa tempo detection
        tempo_librosa, beats_librosa = librosa.beat.beat_track(y=y, sr=sr)

        # madmom advanced beat tracking (more accurate)
        beat_processor = RNNBeatProcessor()
        beats_madmom = beat_processor(str(audio_path))

        # madmom tempo estimation with confidence
        tempo_processor = TempoEstimationProcessor(fps=100)
        tempo_madmom = tempo_processor(beats_madmom)

        # Use madmom results (generally more accurate) with librosa fallback
        tempo = float(tempo_madmom[0][0] if len(tempo_madmom) > 0 else tempo_librosa)
        beats = beats_madmom if len(beats_madmom) > 0 else beats_librosa

        # Onset detection for rhythm analysis
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

        # Swing ratio (measure timing between consecutive beats)
        swing_ratio = self._calculate_swing(beats, sr)

        # Velocity estimation (RMS energy per beat)
        velocities = []
        beat_frames = librosa.time_to_frames(beats, sr=sr)
        for i in range(len(beat_frames) - 1):
            start_frame = beat_frames[i]
            end_frame = beat_frames[i + 1]
            segment = y[start_frame:end_frame]
            rms = librosa.feature.rms(y=segment).mean()
            velocities.append(rms)

        # Syncopation (off-beat accents)
        syncopation_index = self._calculate_syncopation(onsets, beats, sr)

        # Rhythmic complexity (onset density + variation)
        complexity = len(onsets) / (len(beats) + 1) * np.std(velocities)

        return {
            'tempo_bpm': float(tempo),
            'tempo_confidence': float(tempo_madmom[0][1]) if len(tempo_madmom) > 0 else 0.7,
            'swing_ratio': swing_ratio,
            'syncopation_index': syncopation_index,
            'velocity_mean': float(np.mean(velocities)),
            'velocity_std': float(np.std(velocities)),
            'hit_density': float(len(onsets) / len(beats)),
            'rhythmic_complexity': float(complexity),
            'confidence': 0.85  # High confidence with madmom + librosa
        }
```

---

#### 3.4 MIDI Database Collector
**Location:** `src/research/collectors/midi_db.py`

**Sources:**
- BitMIDI: `https://bitmidi.com/search?q={artist}`
- FreeMIDI: `https://freemidi.org/search/{artist}`
- Musescore: `https://musescore.com/sheetmusic?text={artist}`

**Implementation:**
```python
class MidiDatabaseCollector(BaseCollector):
    async def collect(self, artist_name: str) -> List[ResearchSource]:
        midi_files = []

        # Search all MIDI databases
        for db in [BitMIDI(), FreeMIDI(), Musescore()]:
            results = await db.search(artist_name)
            midi_files.extend(results)

        # Download and analyze MIDI files
        analyzed = []
        for midi_url in midi_files[:10]:  # Limit to 10 files
            try:
                midi_path = await self._download_midi(midi_url)

                # Extract drum track
                drum_track = self._extract_drum_track(midi_path)

                if drum_track:
                    # Analyze patterns
                    patterns = self._analyze_patterns(drum_track)

                    analyzed.append(ResearchSource(
                        source_type='midi',
                        url=midi_url,
                        file_path=str(midi_path),
                        extracted_data=patterns,
                        confidence=0.9  # High confidence for MIDI
                    ))
            except Exception as e:
                logger.warning(f"MIDI analysis failed for {midi_url}: {e}")

        return analyzed

    def _extract_drum_track(self, midi_path: Path) -> Optional[MidiTrack]:
        """Find and extract drum track (channel 10 or drum program)."""
        mid = MidiFile(midi_path)

        for track in mid.tracks:
            # Check for channel 10 (GM drums)
            is_drums = any(
                msg.channel == 9 for msg in track
                if hasattr(msg, 'channel')
            )

            if is_drums:
                return track

        return None

    def _analyze_patterns(self, track: MidiTrack) -> dict:
        """Extract rhythm patterns from MIDI track."""
        patterns = {
            'kick': [],
            'snare': [],
            'hihat': [],
            'tempo': None,
            'time_signature': (4, 4)
        }

        current_time = 0
        for msg in track:
            current_time += msg.time

            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.note in [35, 36]:  # Kick
                    patterns['kick'].append({
                        'time': current_time,
                        'velocity': msg.velocity
                    })
                elif msg.note in [38, 40]:  # Snare
                    patterns['snare'].append({
                        'time': current_time,
                        'velocity': msg.velocity
                    })
                elif msg.note in [42, 44, 46]:  # Hi-hat
                    patterns['hihat'].append({
                        'time': current_time,
                        'velocity': msg.velocity
                    })

        return patterns
```

---

### 4. Style Profile Builder
**Location:** `src/research/profile_builder.py`

**Responsibilities:**
- Aggregate data from all collectors
- Resolve conflicts (e.g., different tempo values)
- Extract quantitative parameters
- Generate text description for LLM
- Create vector embedding
- Calculate confidence score

**Implementation:**
```python
class StyleProfileBuilder:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    async def build(
        self,
        artist_name: str,
        collector_results: List[List[ResearchSource]]
    ) -> StyleProfile:
        # Flatten results
        all_sources = [
            source for results in collector_results
            for source in results if isinstance(results, list)
        ]

        # Extract parameters
        tempo_params = self._extract_tempo(all_sources)
        swing_params = self._extract_swing(all_sources)
        velocity_params = self._extract_velocity(all_sources)

        # Generate text description
        text_desc = self._generate_description(
            artist_name,
            all_sources,
            tempo_params,
            swing_params
        )

        # Create embedding
        embedding = self.embedder.encode(text_desc)

        # Calculate confidence
        confidence = self._calculate_confidence(all_sources)

        # Extract MIDI templates
        midi_templates = [
            source.file_path for source in all_sources
            if source.source_type == 'midi' and source.file_path
        ]

        return StyleProfile(
            artist_name=artist_name,
            text_description=text_desc,
            quantitative_params={
                'tempo_min': tempo_params['min'],
                'tempo_max': tempo_params['max'],
                'tempo_avg': tempo_params['avg'],
                'swing_percent': swing_params['avg'],
                'ghost_note_prob': self._estimate_ghost_notes(all_sources),
                'syncopation_level': self._estimate_syncopation(all_sources),
                'velocity_mean': velocity_params['mean'],
                'velocity_std': velocity_params['std']
            },
            midi_templates=midi_templates,
            embedding=embedding,
            confidence_score=confidence,
            sources_count={
                'papers': sum(1 for s in all_sources if s.source_type == 'paper'),
                'articles': sum(1 for s in all_sources if s.source_type == 'article'),
                'audio': sum(1 for s in all_sources if s.source_type == 'audio'),
                'midi': sum(1 for s in all_sources if s.source_type == 'midi')
            }
        )

    def _extract_tempo(self, sources: List[ResearchSource]) -> dict:
        """Aggregate tempo data from all sources."""
        tempos = []

        for source in sources:
            if source.source_type == 'audio':
                tempos.append(source.extracted_data['tempo_bpm'])
            elif source.source_type == 'midi':
                if source.extracted_data.get('tempo'):
                    tempos.append(source.extracted_data['tempo'])
            elif source.source_type in ['paper', 'article']:
                # Extract tempo mentions from text using regex
                text = source.raw_content
                tempo_matches = re.findall(r'(\d+)\s*BPM', text, re.IGNORECASE)
                tempos.extend([int(t) for t in tempo_matches])

        if not tempos:
            return {'min': 120, 'max': 120, 'avg': 120}

        # Filter outliers (remove values > 2 std dev from mean)
        tempos = np.array(tempos)
        mean = tempos.mean()
        std = tempos.std()
        filtered = tempos[np.abs(tempos - mean) <= 2 * std]

        return {
            'min': int(filtered.min()),
            'max': int(filtered.max()),
            'avg': int(filtered.mean())
        }

    def _generate_description(
        self,
        artist_name: str,
        sources: List[ResearchSource],
        tempo: dict,
        swing: dict
    ) -> str:
        """Generate text description for LLM prompt."""
        # Extract key phrases from articles/papers
        key_phrases = []
        for source in sources:
            if source.source_type in ['paper', 'article']:
                # Simple extraction (in production, use more sophisticated NLP)
                phrases = self._extract_style_phrases(source.raw_content)
                key_phrases.extend(phrases)

        # Build description
        description = f"{artist_name} is known for "

        if key_phrases:
            description += f"{', '.join(key_phrases[:3])}. "

        description += f"Typical tempo range: {tempo['min']}-{tempo['max']} BPM "
        description += f"(average {tempo['avg']} BPM). "

        if swing['avg'] > 55:
            description += f"Swing feel at {swing['avg']:.0f}%. "
        else:
            description += "Straight timing. "

        return description
```

---

### 5. Generation Layer

#### 5.1 LLM Provider Manager
**Location:** `src/generation/providers/manager.py`

**Supported Providers:**
- Anthropic (Claude-3.5-sonnet) - **Primary**
- Google (Gemini-2.5/3) - **Secondary**
- OpenAI (ChatGPT-5.1, GPT-4) - **Tertiary/Fallback**

**Implementation:**
```python
class LLMProviderManager:
    def __init__(self, config: dict):
        self.providers = {}
        self.primary = config['primary_provider']
        self.fallbacks = config['fallback_providers']

        # Initialize providers
        if 'openai' in config:
            self.providers['openai'] = OpenAIProvider(
                api_key=os.getenv(config['openai']['api_key_env']),
                model=config['openai']['model']
            )

        if 'anthropic' in config:
            self.providers['anthropic'] = AnthropicProvider(
                api_key=os.getenv(config['anthropic']['api_key_env']),
                model=config['anthropic']['model']
            )

        if 'google' in config:
            self.providers['google'] = GoogleProvider(
                api_key=os.getenv(config['google']['api_key_env']),
                model=config['google']['model']
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        provider_name: Optional[str] = None
    ) -> dict:
        """Generate with fallback logic."""
        provider_name = provider_name or self.primary
        providers_to_try = [provider_name] + [
            p for p in self.fallbacks if p != provider_name
        ]

        for provider in providers_to_try:
            try:
                result = await self.providers[provider].generate_midi_tokens(
                    prompt=prompt,
                    system_prompt=system_prompt
                )
                logger.info(f"Generation successful with {provider}")
                return {'result': result, 'provider_used': provider}
            except Exception as e:
                logger.warning(f"{provider} failed: {e}")
                continue

        raise Exception("All LLM providers failed")
```

---

#### 5.2 Prompt Engineering
**Location:** `src/generation/prompt_builder.py`

**System Prompt Template:**
```python
SYSTEM_PROMPT = """You are an expert AI drummer specializing in authentic recreation of drumming styles.

Your task is to generate drum MIDI patterns as structured JSON data.

OUTPUT FORMAT (strict):
{
  "notes": [
    {
      "pitch": 36,      # MIDI note number (35-81 for drums)
      "velocity": 90,   # 1-127
      "time": 0,        # Position in ticks (480 per quarter note)
      "duration": 120   # Note length in ticks
    }
  ],
  "tempo": 120,
  "time_signature": [4, 4],
  "total_bars": 4
}

DRUM MAPPING (GM Standard):
- 35/36: Bass Drum (Kick)
- 38/40: Snare
- 42: Closed Hi-Hat
- 44: Pedal Hi-Hat
- 46: Open Hi-Hat
- 49/57: Crash Cymbal
- 51: Ride Cymbal
- 53: Ride Bell

RULES:
1. Output ONLY valid JSON (no explanations)
2. Time values must align to grid (multiples of 30 for 32nd notes)
3. Velocity: 1-127 (typical range: 40-120)
4. Duration: minimum 30 ticks
5. Pattern must be musically coherent
6. Include realistic velocity variation
7. Follow the artist's documented style characteristics
"""

def build_user_prompt(
    style_profile: StyleProfile,
    bars: int,
    tempo: int,
    time_signature: Tuple[int, int]
) -> str:
    prompt = f"""
Generate {bars} bars of drum MIDI in the style of {style_profile.artist_name}.

STYLE CHARACTERISTICS:
{style_profile.text_description}

QUANTITATIVE PARAMETERS:
- Tempo: {tempo} BPM (artist typical: {style_profile.quantitative_params['tempo_min']}-{style_profile.quantitative_params['tempo_max']} BPM)
- Swing: {style_profile.quantitative_params['swing_percent']:.1f}%
- Syncopation: {style_profile.quantitative_params['syncopation_level']:.2f}
- Ghost note probability: {style_profile.quantitative_params['ghost_note_prob']:.2f}
- Average velocity: {style_profile.quantitative_params['velocity_mean']:.0f}

SIGNATURE PATTERNS (reference):
{_format_midi_templates(style_profile.midi_templates[:2])}

Generate {bars} bars at {tempo} BPM, {time_signature[0]}/{time_signature[1]} time.

Output JSON only, no explanation.
"""
    return prompt
```

---

#### 5.3 Template-Based Generator
**Location:** `src/generation/template_generator.py`

**Used as fallback when LLM fails or for augmentation.**

```python
class TemplateGenerator:
    def generate(
        self,
        style_profile: StyleProfile,
        params: GenerationParams
    ) -> dict:
        """Generate using MIDI templates with variations."""
        # Load templates
        templates = self._load_templates(style_profile.midi_templates)

        if not templates:
            # Fallback to generic patterns
            return self._generate_generic(params)

        # Select random template
        template = random.choice(templates)

        # Extract patterns
        patterns = self._extract_patterns(template)

        # Apply style-specific variations
        patterns = self._apply_swing(
            patterns,
            style_profile.quantitative_params['swing_percent']
        )
        patterns = self._add_ghost_notes(
            patterns,
            style_profile.quantitative_params['ghost_note_prob']
        )
        patterns = self._vary_velocities(
            patterns,
            style_profile.quantitative_params['velocity_mean'],
            style_profile.quantitative_params['velocity_std']
        )

        # Convert to JSON format
        midi_json = self._patterns_to_json(patterns, params)

        return midi_json
```

---

## Data Flow

### Flow 1: First-Time Artist (Research → Generation)

```
User Input: "John Bonham"
       ↓
[Orchestrator] Check cache
       ↓
   Cache MISS
       ↓
[Research Orchestrator] Delegate research
       ↓
   ┌───┴───┐
   │       │
[Papers] [Articles] [Audio] [MIDI]  ← Parallel execution
   │       │       │      │
   └───┬───┘       │      │
       └───────────┴──────┘
              ↓
    [Style Profile Builder]
    - Aggregate data
    - Extract parameters
    - Generate description
    - Create embedding
              ↓
      [Database] Save profile
              ↓
[Orchestrator] Continue to generation
              ↓
  [LLM Provider Manager]
  - Try Claude 3.5 Sonnet → Success!
              ↓
  [MIDI Export & Validation]
  - Convert JSON → MIDI
  - Apply humanization
  - Generate 4 variations
              ↓
  [File Storage] Save MIDI files
              ↓
  [API Response] Return file paths
              ↓
  [Max for Live] Import clips
```

**Total Time:** 5-20 minutes (research) + < 2 minutes (generation)

---

### Flow 2: Cached Artist (Direct Generation)

```
User Input: "John Bonham"
       ↓
[Orchestrator] Check cache
       ↓
   Cache HIT
       ↓
[Database] Load StyleProfile
       ↓
[LLM Provider Manager]
  - Build prompt with profile
  - Try Claude 3.5 Sonnet → Success!
       ↓
[MIDI Export & Validation]
       ↓
[File Storage]
       ↓
[API Response]
       ↓
[Max for Live] Import clips
```

**Total Time:** < 2 minutes

---

## Technology Stack

### Backend (Python 3.11+)

```python
# requirements.txt

# Web Framework
fastapi==0.109.2
uvicorn[standard]==0.27.1
pydantic==2.6.1
python-multipart==0.0.9

# Async Task Queue
celery==5.3.6
redis==5.0.1
flower==2.0.1  # Celery monitoring

# Database
sqlalchemy==2.0.25
alembic==1.13.1
psycopg2-binary==2.9.9
asyncpg==0.29.0
pgvector==0.2.4

# LLM Providers
openai==1.12.0
anthropic==0.18.1
google-generativeai==0.3.2

# Web Scraping & Research
beautifulsoup4==4.12.3
scrapy==2.11.1
newspaper3k==0.2.8
requests==2.31.0
aiohttp==3.9.3
httpx==0.26.0

# Audio Processing
librosa==0.10.2.post1
madmom==0.16.1            # Advanced beat/tempo tracking (cross-platform)
yt-dlp==2024.12.13
soundfile==0.12.1
audioread==3.0.1

# NLP & Embeddings
spacy==3.7.4
sentence-transformers==2.3.1
transformers==4.37.2

# MIDI Processing
mido==1.3.2
python-rtmidi==1.5.8

# Scientific Computing
numpy==1.26.4
scipy==1.12.0
pandas==2.2.0

# Utilities
python-dotenv==1.0.1
pyyaml==6.0.1
loguru==0.7.2
click==8.1.7
tqdm==4.66.1

# Testing
pytest==8.0.0
pytest-asyncio==0.23.5
pytest-cov==4.1.0
httpx==0.26.0  # For FastAPI testing

# Development
black==24.1.1
ruff==0.2.1
mypy==1.8.0
pre-commit==3.6.0
```

### Database
- **PostgreSQL 15.5+** with **pgvector 0.5.1** extension
- Connection pooling with SQLAlchemy
- Async support via asyncpg

### Cache & Queue
- **Redis 7.2.4** for:
  - Celery task queue
  - Rate limiting
  - Response caching
  - Session storage

### Frontend (Max for Live)
- **Max 8.5.8** or later
- JavaScript ES6+ for HTTP bridge
- Live API 12.0+ for clip manipulation

---

## Database Schema

### PostgreSQL Schema

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Artists table
CREATE TABLE artists (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    research_status VARCHAR(50) DEFAULT 'pending',
    -- Status: pending, researching, cached, failed
    last_updated TIMESTAMP DEFAULT NOW(),
    sources_count INTEGER DEFAULT 0,
    confidence_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_artist_name (name),
    INDEX idx_research_status (research_status)
);

-- Research sources table
CREATE TABLE research_sources (
    id SERIAL PRIMARY KEY,
    artist_id INTEGER REFERENCES artists(id) ON DELETE CASCADE,
    source_type VARCHAR(50) NOT NULL,
    -- Types: paper, article, audio, midi
    url TEXT,
    file_path TEXT,
    raw_content TEXT,
    extracted_data JSONB,
    confidence REAL DEFAULT 0.5,
    collected_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_artist_id (artist_id),
    INDEX idx_source_type (source_type)
);

-- Style profiles table
CREATE TABLE style_profiles (
    id SERIAL PRIMARY KEY,
    artist_id INTEGER UNIQUE REFERENCES artists(id) ON DELETE CASCADE,
    text_description TEXT NOT NULL,
    quantitative_params JSONB NOT NULL,
    -- JSON structure:
    -- {
    --   "tempo_min": int,
    --   "tempo_max": int,
    --   "tempo_avg": int,
    --   "swing_percent": float,
    --   "ghost_note_prob": float,
    --   "syncopation_level": float,
    --   "velocity_mean": int,
    --   "velocity_std": int
    -- }
    midi_templates_json JSONB,
    -- Array of file paths
    embedding vector(384),
    -- Sentence-transformers dimension
    confidence_score REAL NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_profile_artist_id (artist_id)
);

-- Create vector index for similarity search
CREATE INDEX ON style_profiles USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Generation history table
CREATE TABLE generation_history (
    id SERIAL PRIMARY KEY,
    artist_id INTEGER REFERENCES artists(id) ON DELETE SET NULL,
    provider_used VARCHAR(50),
    -- openai, anthropic, google, template
    generation_time_ms INTEGER,
    user_params JSONB,
    -- {
    --   "bars": int,
    --   "tempo": int,
    --   "time_signature": [int, int],
    --   "variations": int
    -- }
    output_files JSONB,
    -- Array of file paths
    created_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_gen_artist_id (artist_id),
    INDEX idx_gen_created_at (created_at)
);

-- User sessions (future)
CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_ip VARCHAR(45),
    requests_count INTEGER DEFAULT 0,
    last_request TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),

    INDEX idx_session_id (session_id)
);
```

### Alembic Migrations

```python
# migrations/versions/001_initial_schema.py

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

def upgrade():
    # Enable pgvector
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Create artists table
    op.create_table(
        'artists',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(255), unique=True, nullable=False),
        sa.Column('research_status', sa.String(50), default='pending'),
        sa.Column('last_updated', sa.DateTime(), default=sa.func.now()),
        sa.Column('sources_count', sa.Integer(), default=0),
        sa.Column('confidence_score', sa.Float(), default=0.0),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now())
    )

    # ... (rest of schema)
```

---

## API Specification

### REST API Endpoints

#### Research Endpoints

**POST /api/v1/research**
```json
Request:
{
  "artist": "John Bonham",
  "depth": "full"  // Optional: "quick" or "full"
}

Response (202 Accepted):
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "researching",
  "estimated_time_minutes": 15
}
```

**GET /api/v1/research/{artist}**
```json
Response (200 OK):
{
  "exists": true,
  "confidence": 0.85,
  "last_updated": "2025-11-15T10:30:00Z",
  "sources_count": {
    "papers": 5,
    "articles": 12,
    "audio": 3,
    "midi": 4
  }
}

Response (404 Not Found):
{
  "exists": false,
  "message": "Artist not found in cache"
}
```

**POST /api/v1/augment/{artist}**
```json
Request:
{}

Response (202 Accepted):
{
  "task_id": "...",
  "status": "augmenting",
  "current_sources": 24,
  "target_sources": 30
}
```

---

#### Generation Endpoints

**POST /api/v1/generate**
```json
Request:
{
  "artist": "John Bonham",
  "bars": 4,
  "tempo": 120,
  "time_signature": [4, 4],
  "variations": 4,
  "provider": "auto",  // Optional: "openai", "anthropic", "google", "auto"
  "humanize": true
}

Response (200 OK):
{
  "status": "success",
  "generation_time_ms": 1847,
  "provider_used": "openai",
  "midi_files": [
    "/output/john_bonham_var1_20251117_103045.mid",
    "/output/john_bonham_var2_20251117_103045.mid",
    "/output/john_bonham_var3_20251117_103045.mid",
    "/output/john_bonham_var4_20251117_103045.mid"
  ],
  "confidence": 0.85
}

Response (400 Bad Request):
{
  "error": "Artist not researched",
  "message": "Please research this artist first",
  "suggest_endpoint": "/api/v1/research"
}

Response (500 Internal Server Error):
{
  "error": "Generation failed",
  "message": "All LLM providers failed",
  "details": "..."
}
```

---

#### Utility Endpoints

**GET /api/v1/task/{task_id}**
```json
Response (200 OK - In Progress):
{
  "task_id": "...",
  "status": "in_progress",
  "progress": 65,
  "current_step": "Analyzing audio...",
  "estimated_completion": "2025-11-17T10:45:00Z"
}

Response (200 OK - Completed):
{
  "task_id": "...",
  "status": "completed",
  "progress": 100,
  "result": {
    "artist_id": 123,
    "confidence": 0.82
  }
}

Response (200 OK - Failed):
{
  "task_id": "...",
  "status": "failed",
  "error": "Insufficient data found",
  "details": "Only 2 sources found, minimum 5 required"
}
```

**GET /api/v1/artists**
```json
Response (200 OK):
{
  "total": 1523,
  "cached": 1523,
  "researching": 0,
  "failed": 0,
  "recent": [
    {"name": "John Bonham", "confidence": 0.89, "last_updated": "..."},
    {"name": "Travis Barker", "confidence": 0.76, "last_updated": "..."}
  ]
}
```

**GET /api/v1/similar/{artist}?limit=5**
```json
Response (200 OK):
{
  "artist": "John Bonham",
  "similar_artists": [
    {"name": "Keith Moon", "similarity": 0.91},
    {"name": "Ginger Baker", "similarity": 0.85},
    {"name": "Neil Peart", "similarity": 0.78}
  ]
}
```

---

## Deployment Architecture

### Development Environment

```
┌──────────────────────────────────────┐
│         Developer Machine            │
│                                      │
│  ┌────────────┐  ┌────────────────┐ │
│  │  FastAPI   │  │  PostgreSQL    │ │
│  │  (localhost│  │  (Docker)      │ │
│  │   :8000)   │  │  (port 5432)   │ │
│  └────────────┘  └────────────────┘ │
│                                      │
│  ┌────────────┐  ┌────────────────┐ │
│  │   Redis    │  │  Celery Worker │ │
│  │  (Docker)  │  │  (local)       │ │
│  │ (port 6379)│  │                │ │
│  └────────────┘  └────────────────┘ │
│                                      │
│  ┌────────────────────────────────┐ │
│  │     Ableton Live + M4L         │ │
│  │  (calls localhost:8000)        │ │
│  └────────────────────────────────┘ │
└──────────────────────────────────────┘
```

**Setup:**
```bash
# Docker Compose for dependencies
docker-compose up -d postgres redis

# Run FastAPI
uvicorn src.api.main:app --reload --port 8000

# Run Celery worker
celery -A src.tasks.worker worker --loglevel=info

# Run tests
pytest tests/
```

---

### Production Environment

```
┌────────────────────────────────────────────────────────────────┐
│                        Load Balancer (Nginx)                   │
│                         (HTTPS/SSL)                            │
└────────────┬──────────────────────────────┬────────────────────┘
             │                              │
             ▼                              ▼
┌─────────────────────────┐    ┌─────────────────────────┐
│   FastAPI Instance 1    │    │   FastAPI Instance 2    │
│   (Docker Container)    │    │   (Docker Container)    │
└────────────┬────────────┘    └────────────┬────────────┘
             │                              │
             └──────────────┬───────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
    ┌────────────────────┐  ┌────────────────────┐
    │  PostgreSQL (RDS)  │  │  Redis (Elastica Cache) │
    │  - Primary         │  │  - Task Queue      │
    │  - Read Replica    │  │  - Cache           │
    └────────────────────┘  └────────────────────┘
                │
                │
                ▼
    ┌────────────────────────────┐
    │   Celery Workers (ECS)     │
    │   - Research tasks         │
    │   - Generation tasks       │
    │   (Auto-scaling 2-10)      │
    └────────────────────────────┘
                │
                ▼
    ┌────────────────────────────┐
    │   S3 Bucket                │
    │   - Generated MIDI files   │
    │   - MIDI templates         │
    └────────────────────────────┘
```

**Infrastructure:**
- **Cloud Provider:** AWS (or equivalent)
- **API:** ECS Fargate (2-4 containers)
- **Workers:** ECS Fargate (2-10 containers, auto-scaling)
- **Database:** RDS PostgreSQL (db.t3.medium)
- **Cache:** ElastiCache Redis (cache.t3.small)
- **Storage:** S3 for MIDI files
- **CDN:** CloudFront for MIDI file delivery
- **Monitoring:** CloudWatch + Prometheus + Grafana

---

## Security

### API Security

**1. API Key Authentication**
```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

**2. Rate Limiting**
```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/api/v1/generate")
@limiter(times=10, seconds=60)  # 10 requests per minute
async def generate_pattern(...):
    pass
```

**3. Input Validation**
```python
from pydantic import BaseModel, validator

class GenerateRequest(BaseModel):
    artist: str
    bars: int
    tempo: int

    @validator('artist')
    def validate_artist(cls, v):
        if len(v) > 100:
            raise ValueError('Artist name too long')
        if not re.match(r'^[a-zA-Z0-9\s\-]+$', v):
            raise ValueError('Invalid characters in artist name')
        return v

    @validator('bars')
    def validate_bars(cls, v):
        if not 1 <= v <= 16:
            raise ValueError('Bars must be 1-16')
        return v
```

**4. CORS Configuration**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*"],  # Dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Data Security

**1. Environment Variables**
```bash
# .env (never commit this!)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
DATABASE_URL=postgresql://user:pass@localhost/db
SECRET_KEY=...
```

**2. Database Encryption**
- SSL/TLS for database connections
- Encrypt sensitive fields (API keys in DB)
- Regular backups with encryption

**3. LLM API Keys**
- Store in environment variables
- Rotate keys quarterly
- Monitor usage for anomalies

---

## Performance Optimization

### Database Optimization

**1. Indexing Strategy**
```sql
-- Indexes on frequently queried columns
CREATE INDEX idx_artist_name ON artists(name);
CREATE INDEX idx_research_status ON artists(research_status);
CREATE INDEX idx_gen_created_at ON generation_history(created_at);

-- Vector index for similarity search
CREATE INDEX ON style_profiles USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

**2. Query Optimization**
```python
# Use select_related / joinedload for related data
artists = session.query(Artist).options(
    joinedload(Artist.style_profile),
    joinedload(Artist.research_sources)
).filter(Artist.name == name).first()
```

**3. Connection Pooling**
```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

### Caching Strategy

**1. Response Caching**
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

@app.get("/api/v1/artists")
@cache(expire=3600)  # Cache for 1 hour
async def list_artists():
    return await db.get_all_artists()
```

**2. StyleProfile Caching**
```python
# Cache in Redis for fast access
async def get_profile(artist_name: str) -> StyleProfile:
    # Try Redis first
    cached = await redis.get(f"profile:{artist_name}")
    if cached:
        return StyleProfile.parse_raw(cached)

    # Fallback to DB
    profile = await db.get_profile(artist_name)

    # Cache for 7 days
    await redis.setex(
        f"profile:{artist_name}",
        604800,
        profile.json()
    )

    return profile
```

### Async Processing

**All I/O operations use async/await:**
```python
async def research_artist(artist_name: str):
    # Run collectors in parallel
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_papers(session, artist_name),
            fetch_articles(session, artist_name),
            analyze_audio(artist_name),
            search_midi(artist_name)
        ]
        results = await asyncio.gather(*tasks)
    return results
```

---

## Monitoring & Observability

### Logging

**Structured Logging with Loguru:**
```python
from loguru import logger

logger.add(
    "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time} | {level} | {message}",
    serialize=True  # JSON format
)

logger.info("Artist researched", artist="John Bonham", sources=24, confidence=0.85)
```

### Metrics

**Prometheus Metrics:**
```python
from prometheus_client import Counter, Histogram, Gauge

# Request counters
requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

# Generation time
generation_duration = Histogram(
    'generation_duration_seconds',
    'Time spent generating patterns',
    ['provider']
)

# Cached artists gauge
cached_artists = Gauge(
    'cached_artists_total',
    'Number of cached artists'
)
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    checks = {
        "api": "healthy",
        "database": await check_database(),
        "redis": await check_redis(),
        "celery": await check_celery()
    }

    all_healthy = all(v == "healthy" for v in checks.values())
    status_code = 200 if all_healthy else 503

    return JSONResponse(checks, status_code=status_code)
```

---

## Error Handling

### Error Response Format

```json
{
  "error": "generation_failed",
  "message": "All LLM providers failed to generate pattern",
  "details": {
    "openai": "Rate limit exceeded",
    "anthropic": "API key invalid",
    "google": "Connection timeout"
  },
  "timestamp": "2025-11-17T10:30:00Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Exception Hierarchy

```python
class MidiDrumiGenError(Exception):
    """Base exception."""
    pass

class ResearchError(MidiDrumiGenError):
    """Research pipeline failed."""
    pass

class GenerationError(MidiDrumiGenError):
    """Generation failed."""
    pass

class LLMProviderError(GenerationError):
    """LLM provider failure."""
    pass

class DatabaseError(MidiDrumiGenError):
    """Database operation failed."""
    pass
```

---

## Appendix: File Structure

```
MidiDrumiGen/
├── src/
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── research/
│   │   ├── __init__.py
│   │   ├── orchestrator.py
│   │   ├── profile_builder.py
│   │   └── collectors/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── papers.py
│   │       ├── articles.py
│   │       ├── audio.py
│   │       └── midi_db.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── openai.py
│   │   │   ├── anthropic.py
│   │   │   ├── google.py
│   │   │   └── manager.py
│   │   ├── prompt_builder.py
│   │   ├── template_generator.py
│   │   └── hybrid_coordinator.py
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── manager.py
│   │   └── migrations/
│   │       └── versions/
│   ├── midi/
│   │   ├── __init__.py
│   │   └── export_and_validate.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── routes/
│   │       ├── research.py
│   │       ├── generate.py
│   │       └── utils.py
│   ├── tasks/
│   │   ├── __init__.py
│   │   └── worker.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── validation.py
├── ableton/
│   ├── MidiDrumGen.amxd
│   └── js/
│       └── bridge.js
├── configs/
│   └── generation.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── ARCHITECTURE.md (this file)
│   ├── PRD.md
│   ├── ORCHESTRATOR_META_PROMPT.md
│   └── UI.md
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

**Document Version:** 1.0
**Last Updated:** 2025-11-17
**Next Review:** 2025-12-01
