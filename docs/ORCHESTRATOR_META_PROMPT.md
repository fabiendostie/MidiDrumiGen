# META PROMPT: MidiDrumiGen - On-Demand Artist-Style MIDI Generation System

**Version:** 2.0
**Date:** 2025-11-17
**Target:** Claude Code CLI / AI Development Agents

---

## Project Context

You are implementing a revolutionary MIDI drum pattern generation plugin for Ableton Live. The system allows users to input ANY artist/band name and receive drum patterns generated in that artist's authentic style within 2 minutes (for cached artists) or 5-20 minutes for first-time research.

**Critical Constraints:**
- This is MIDI-only (no audio generation)
- Must support any artist name (not pre-trained models)
- Research happens on-demand with caching
- Generation must be < 2 minutes for cached artists
- Multi-LLM provider support (OpenAI, Claude, Gemini, future: local LLMs)
- Final output: Max for Live device in Ableton

---

## System Architecture: Orchestrator + Sub-Agents

### ORCHESTRATOR AGENT
**Role:** Coordinates all sub-agents, manages workflow, handles caching decisions

**Responsibilities:**
1. Receive artist name from user
2. Check database cache for existing StyleProfile
3. If cached: Route to Generation Agent
4. If not cached: Delegate to Research Orchestrator Sub-Agent
5. Monitor progress and update UI
6. Handle errors and fallbacks
7. Manage database transactions

**Implementation:** `src/orchestrator/main.py`

**Key Methods:**
```python
async def process_request(artist_name: str, params: dict) -> dict:
    # Check cache
    # Route to research or generation
    # Monitor progress
    # Return results
```

---

### SUB-AGENT 1: Research Orchestrator
**Role:** Coordinates data collection from multiple sources

**Sub-Agents Under Research Orchestrator:**

#### 1a. Scholar Paper Collector
**Implementation:** `src/research/collectors/papers.py`

**Data Sources:**
- Semantic Scholar API (https://api.semanticscholar.org/)
- arXiv API (https://arxiv.org/help/api/)
- CrossRef API (https://www.crossref.org/documentation/)

**Search Terms:**
- "[artist] drumming analysis"
- "[artist] rhythm characteristics"
- "[band] musical style analysis"
- "[drummer] technique study"

**Extraction:**
- Abstract and conclusion text
- Tempo mentions (regex: `\d+\s*BPM`)
- Groove descriptors (swing, syncopation, etc.)
- Citation count (for quality ranking)

**Output:**
```python
{
    "source_type": "paper",
    "title": str,
    "authors": List[str],
    "abstract": str,
    "extracted_tempo": Optional[int],
    "keywords": List[str],
    "confidence": float
}
```

---

#### 1b. Web Article Collector
**Implementation:** `src/research/collectors/articles.py`

**Data Sources:**
- Music journalism sites (Pitchfork, Rolling Stone, Drummerworld)
- Wikipedia (DBpedia structured data)
- Genius (lyrics + annotations about drumming)

**Technologies:**
- BeautifulSoup4 4.12+ for HTML parsing
- Scrapy 2.11+ for crawling
- Newspaper3k 0.2.8 for article extraction
- spaCy 3.7+ for NLP

**Extraction:**
- Interview quotes about drumming
- Style descriptions from reviews
- Equipment mentions (drum kits, cymbals)
- Collaborator names (find similar artists)

**Output:**
```python
{
    "source_type": "article",
    "url": str,
    "title": str,
    "publication": str,
    "text_content": str,
    "drumming_mentions": List[str],
    "confidence": float
}
```

---

#### 1c. Audio Analysis Collector
**Implementation:** `src/research/collectors/audio.py`

**Data Sources:**
- YouTube (via yt-dlp)
- SoundCloud API
- Spotify API (metadata only)

**Audio Processing:**
- Librosa 0.10.2+ for tempo/beat detection, rhythm analysis
- madmom 0.16.1 for advanced beat/tempo tracking (100% cross-platform, replaces Essentia)

**Extracted Features:**
- Tempo (BPM) with confidence
- Beat positions (for swing calculation)
- Onset detection (for syncopation analysis)
- Spectral features (for instrument identification)
- Velocity estimation (RMS energy per beat)

**Output:**
```python
{
    "source_type": "audio",
    "video_id": str,
    "title": str,
    "tempo_bpm": float,
    "swing_ratio": float,  # 50-67%
    "syncopation_index": float,  # 0-1
    "hit_density": float,  # notes per beat
    "velocity_distribution": dict,  # histogram
    "confidence": float
}
```

---

#### 1d. MIDI Database Collector
**Implementation:** `src/research/collectors/midi_db.py`

**Data Sources:**
- BitMIDI (https://bitmidi.com/)
- FreeMIDI (https://freemidi.org/)
- Musescore (public domain scores)

**Search Strategy:**
- Search by artist/band name
- Filter for drum tracks (channel 10 or drum programs)
- Validate MIDI structure
- Extract pattern templates

**Extraction:**
- Note density per beat
- Velocity patterns
- Kick/snare/hihat rhythms
- Fill patterns
- Groove templates

**Output:**
```python
{
    "source_type": "midi",
    "url": str,
    "file_path": Path,
    "tempo": int,
    "time_signature": Tuple[int, int],
    "pattern_summary": dict,  # kick/snare/hh patterns
    "confidence": float
}
```

---

**Research Orchestrator Coordination:**

**Implementation:** `src/research/orchestrator.py`

```python
class ResearchOrchestrator:
    def __init__(self):
        self.collectors = {
            'papers': ScholarPaperCollector(),
            'articles': WebArticleCollector(),
            'audio': AudioAnalysisCollector(),
            'midi': MidiDatabaseCollector()
        }

    async def research_artist(
        self,
        artist_name: str,
        depth: str = "full"  # "quick" or "full"
    ) -> Dict:
        # Run all collectors in parallel
        tasks = [
            collector.collect(artist_name)
            for collector in self.collectors.values()
        ]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        profile = self.build_profile(results)

        # Store in database
        await self.store_profile(profile)

        return profile
```

---

### SUB-AGENT 2: Style Profile Builder
**Role:** Transform raw research data into generation-ready StyleProfile

**Implementation:** `src/research/profile_builder.py`

**Data Aggregation:**

1. **Tempo Extraction:**
   - Collect all tempo mentions from all sources
   - Weight by source confidence
   - Calculate min/max/avg/mode
   - Filter outliers (>2 std dev)

2. **Swing Calculation:**
   - From audio analysis (primary)
   - From MIDI patterns (secondary)
   - From text ("swung eighths" → 62% swing)

3. **Velocity Distribution:**
   - From audio RMS energy
   - From MIDI velocity values
   - Create histogram (bins: soft/medium/hard)

4. **Pattern Templates:**
   - Extract kick/snare/hihat rhythms from MIDI
   - Identify signature patterns (most frequent)
   - Store as note sequences

5. **Text Description Generation:**
   - Combine descriptive phrases from papers/articles
   - Template: "Known for [style] with [tempo] tempo, characterized by [features]"
   - Used as LLM prompt base

**Output Schema:**
```python
@dataclass
class StyleProfile:
    artist_name: str
    text_description: str  # For LLM prompt
    quantitative_params: dict = field(default_factory=dict)
    # {
    #   "tempo_min": int,
    #   "tempo_max": int,
    #   "tempo_avg": int,
    #   "swing_percent": float,
    #   "ghost_note_prob": float,
    #   "syncopation_level": float,
    #   "velocity_mean": int,
    #   "velocity_std": int
    # }
    midi_templates: List[Path]
    embedding: np.ndarray  # For similarity search
    confidence_score: float  # Overall quality
    sources_count: dict  # Count per source type
    created_at: datetime
    updated_at: datetime
```

**Vector Embedding:**
- Use sentence-transformers (all-MiniLM-L6-v2) for text embedding
- Store in pgvector column
- Enable similarity search for "similar artists"

**Implementation:** `src/research/profile_builder.py`

---

### SUB-AGENT 3: Multi-Provider LLM Manager
**Role:** Abstract LLM provider selection and failover

**Implementation:** `src/generation/providers/`

#### Base Provider Interface
**File:** `src/generation/providers/base.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

class BaseLLMProvider(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    async def generate_midi_tokens(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> Dict:
        """Generate MIDI tokens from prompt."""
        pass

    @abstractmethod
    def validate_credentials(self) -> bool:
        """Check if API key is valid."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass
```

---

#### OpenAI Provider
**File:** `src/generation/providers/openai.py`

```python
from openai import AsyncOpenAI
import json

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate_midi_tokens(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> Dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    @property
    def provider_name(self) -> str:
        return "openai"
```

**Supported Models:**
- gpt-4-turbo-preview (128k context)
- gpt-4-0125-preview
- gpt-3.5-turbo (fallback, cheaper)

---

#### Anthropic Provider
**File:** `src/generation/providers/anthropic.py`

```python
from anthropic import AsyncAnthropic
import json

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate_midi_tokens(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> Dict:
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        content = message.content[0].text
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return json.loads(content)

    @property
    def provider_name(self) -> str:
        return "anthropic"
```

**Supported Models:**
- claude-3-opus-20240229 (best quality)
- claude-3-sonnet-20240229 (balanced)
- claude-3-haiku-20240307 (fast/cheap)

---

#### Google Provider
**File:** `src/generation/providers/google.py`

```python
import google.generativeai as genai
import json

class GoogleProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    async def generate_midi_tokens(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = 2000
    ) -> Dict:
        # Gemini combines system + user prompts
        full_prompt = f"{system_prompt}\n\n{prompt}"

        response = await self.model.generate_content_async(
            full_prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )

        content = response.text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        return json.loads(content)

    @property
    def provider_name(self) -> str:
        return "google"
```

**Supported Models:**
- gemini-1.5-pro (2M token context!)
- gemini-1.5-flash (faster)

---

#### Provider Manager
**File:** `src/generation/providers/manager.py`

```python
class LLMProviderManager:
    def __init__(self, config: dict):
        self.config = config
        self.providers = self._initialize_providers()
        self.primary = config['primary_provider']
        self.fallbacks = config['fallback_providers']

    def _initialize_providers(self) -> Dict[str, BaseLLMProvider]:
        providers = {}

        if 'openai' in self.config:
            providers['openai'] = OpenAIProvider(
                api_key=os.getenv(self.config['openai']['api_key_env']),
                model=self.config['openai']['model']
            )

        # Similar for anthropic, google

        return providers

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        provider_name: Optional[str] = None
    ) -> Dict:
        # Try primary provider
        provider_name = provider_name or self.primary

        try:
            provider = self.providers[provider_name]
            result = await provider.generate_midi_tokens(
                prompt, system_prompt
            )
            return result
        except Exception as e:
            logger.warning(f"{provider_name} failed: {e}")

            # Try fallbacks
            for fallback in self.fallbacks:
                if fallback == provider_name:
                    continue
                try:
                    provider = self.providers[fallback]
                    result = await provider.generate_midi_tokens(
                        prompt, system_prompt
                    )
                    return result
                except Exception as e2:
                    logger.warning(f"{fallback} failed: {e2}")

            raise Exception("All LLM providers failed")
```

---

### SUB-AGENT 4: Prompt Engineering Agent
**Role:** Build optimal prompts for LLM MIDI generation

**Implementation:** `src/generation/prompt_builder.py`

**System Prompt Template:**
```
You are an expert AI drummer specializing in authentic recreation of drumming styles.

Your task is to generate drum MIDI patterns as structured JSON data.

OUTPUT FORMAT (strict):
{
  "notes": [
    {
      "pitch": 36,      # MIDI note number (35-81 for drums)
      "velocity": 90,   # 1-127
      "time": 0,        # Position in ticks (480 per quarter note)
      "duration": 120   # Note length in ticks
    },
    ...
  ],
  "tempo": 120,
  "time_signature": [4, 4],
  "total_bars": 4
}

DRUM MAPPING (GM Standard):
- 35: Acoustic Bass Drum
- 36: Bass Drum 1
- 38: Acoustic Snare
- 40: Electric Snare
- 42: Closed Hi-Hat
- 44: Pedal Hi-Hat
- 46: Open Hi-Hat
- 49: Crash Cymbal 1
- 51: Ride Cymbal 1
- 53: Ride Bell
- 57: Crash Cymbal 2

RULES:
1. Output ONLY valid JSON (no explanations)
2. Time values must align to grid (multiples of 30 for 32nd notes)
3. Velocity: 1-127 (typical range: 40-120)
4. Duration: minimum 30 ticks
5. Pattern must be musically coherent
6. Include realistic velocity variation
```

**User Prompt Template:**
```python
def build_user_prompt(
    style_profile: StyleProfile,
    bars: int = 4,
    tempo: int = 120,
    time_signature: Tuple[int, int] = (4, 4)
) -> str:
    prompt = f"""
Generate {bars} bars of drum MIDI in the style of {style_profile.artist_name}.

STYLE CHARACTERISTICS:
{style_profile.text_description}

QUANTITATIVE PARAMETERS:
- Tempo: {tempo} BPM (artist typical range: {style_profile.quantitative_params['tempo_min']}-{style_profile.quantitative_params['tempo_max']} BPM)
- Swing: {style_profile.quantitative_params['swing_percent']}%
- Syncopation level: {style_profile.quantitative_params['syncopation_level']:.2f}
- Ghost note probability: {style_profile.quantitative_params['ghost_note_prob']:.2f}
- Average velocity: {style_profile.quantitative_params['velocity_mean']}

SIGNATURE PATTERNS:
{_format_midi_templates(style_profile.midi_templates[:2])}

Generate {bars} bars at {tempo} BPM, {time_signature[0]}/{time_signature[1]} time signature.

Output JSON only, no explanation.
"""
    return prompt
```

**Few-Shot Examples:**
Include 1-2 MIDI template examples in JSON format as reference patterns.

---

### SUB-AGENT 5: Template-Based Generator
**Role:** Generate MIDI using pattern templates + rule-based variations

**Implementation:** `src/generation/template_generator.py`

**Algorithm:**

1. **Load Templates:**
   - Select 2-4 MIDI templates from StyleProfile
   - Extract kick, snare, hihat patterns separately

2. **Pattern Extraction:**
```python
def extract_patterns(midi_file: Path) -> Dict[str, List[int]]:
    mid = MidiFile(midi_file)
    patterns = {
        'kick': [],      # Times where kick hits occur
        'snare': [],     # Times where snare hits occur
        'hihat': []      # Times where hihat hits occur
    }

    for msg in mid.tracks[0]:
        if msg.type == 'note_on':
            if msg.note in [35, 36]:  # Kick
                patterns['kick'].append(msg.time)
            elif msg.note in [38, 40]:  # Snare
                patterns['snare'].append(msg.time)
            elif msg.note in [42, 44, 46]:  # Hihat
                patterns['hihat'].append(msg.time)

    return patterns
```

3. **Apply Variations:**
```python
def apply_variations(
    patterns: Dict,
    params: Dict
) -> Dict:
    # Adjust swing
    patterns = apply_swing(patterns, params['swing_percent'])

    # Add ghost notes
    patterns = add_ghost_notes(patterns, params['ghost_note_prob'])

    # Adjust velocities
    patterns = vary_velocities(patterns, params['velocity_std'])

    # Add syncopation
    patterns = add_syncopation(patterns, params['syncopation_level'])

    return patterns
```

4. **Combine Patterns:**
   - Merge kick + snare + hihat into single MIDI track
   - Add fills every 4 bars
   - Ensure no overlapping notes on same pitch

---

### SUB-AGENT 6: Hybrid Generation Coordinator
**Role:** Combine LLM + template approaches for best results

**Implementation:** `src/generation/hybrid_coordinator.py`

**Strategy Flowchart:**
```
1. Try LLM Generation (primary)
   ├─ Success? → Validate output
   │   ├─ Valid? → Return result
   │   └─ Invalid? → Try template generation
   └─ Failed? → Try template generation

2. Template Generation (fallback)
   ├─ Load MIDI templates
   ├─ Apply style transfer
   └─ Return result

3. Post-Processing (both paths)
   ├─ Apply humanization
   ├─ Generate variations (4-8)
   └─ Export MIDI files
```

**Implementation:**
```python
class HybridCoordinator:
    def __init__(
        self,
        llm_manager: LLMProviderManager,
        template_gen: TemplateGenerator
    ):
        self.llm = llm_manager
        self.template_gen = template_gen

    async def generate(
        self,
        style_profile: StyleProfile,
        params: dict
    ) -> List[Path]:
        # Try LLM first
        try:
            result = await self._generate_with_llm(style_profile, params)
            if self._validate_midi(result):
                midi_files = self._create_variations(result, params)
                return midi_files
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")

        # Fallback to templates
        result = self.template_gen.generate(style_profile, params)
        midi_files = self._create_variations(result, params)
        return midi_files

    def _validate_midi(self, midi_data: dict) -> bool:
        # Check JSON structure
        if 'notes' not in midi_data:
            return False

        # Check note ranges
        for note in midi_data['notes']:
            if not (35 <= note['pitch'] <= 81):
                return False
            if not (1 <= note['velocity'] <= 127):
                return False

        return True
```

---

### SUB-AGENT 7: MIDI Export & Validation
**Role:** Convert tokens to MIDI files and validate quality

**Implementation:** `src/midi/export_and_validate.py`

**JSON to MIDI Conversion:**
```python
def json_to_midi(
    midi_data: dict,
    output_path: Path,
    humanize: bool = True
) -> Path:
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    tempo = mido.bpm2tempo(midi_data['tempo'])
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))

    # Set time signature
    ts = midi_data.get('time_signature', [4, 4])
    track.append(mido.MetaMessage(
        'time_signature',
        numerator=ts[0],
        denominator=ts[1]
    ))

    # Sort notes by time
    notes = sorted(midi_data['notes'], key=lambda n: n['time'])

    # Convert to MIDI messages
    current_time = 0
    for note in notes:
        # Note on
        delta_time = note['time'] - current_time
        if humanize:
            delta_time = apply_micro_timing(delta_time)

        velocity = note['velocity']
        if humanize:
            velocity = vary_velocity(velocity)

        track.append(mido.Message(
            'note_on',
            note=note['pitch'],
            velocity=velocity,
            time=delta_time
        ))

        # Note off
        track.append(mido.Message(
            'note_off',
            note=note['pitch'],
            velocity=0,
            time=note['duration']
        ))

        current_time = note['time'] + note['duration']

    mid.save(output_path)
    return output_path
```

**Humanization:**
```python
def apply_micro_timing(time: int, variance_ms: float = 10.0) -> int:
    """Add random timing offset (±variance_ms)."""
    variance_ticks = int((variance_ms / 1000) * 480)  # 480 tpqn
    offset = random.randint(-variance_ticks, variance_ticks)
    return max(0, time + offset)

def vary_velocity(velocity: int, variance: float = 0.1) -> int:
    """Add random velocity variation."""
    offset = int(velocity * random.uniform(-variance, variance))
    return max(1, min(127, velocity + offset))
```

---

### SUB-AGENT 8: Database Manager
**Role:** Handle all database operations

**Implementation:** `src/database/manager.py` + `src/database/models.py`

**Database Schema (SQLAlchemy):**

```python
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Artist(Base):
    __tablename__ = 'artists'

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True, nullable=False)
    research_status = Column(String, default='pending')  # pending/researching/cached/failed
    last_updated = Column(DateTime, default=datetime.utcnow)
    sources_count = Column(Integer, default=0)
    confidence_score = Column(Float, default=0.0)

class ResearchSource(Base):
    __tablename__ = 'research_sources'

    id = Column(Integer, primary_key=True)
    artist_id = Column(Integer, ForeignKey('artists.id'))
    source_type = Column(String)  # paper/article/audio/midi
    url = Column(String, nullable=True)
    file_path = Column(String, nullable=True)
    raw_content = Column(String)
    extracted_data = Column(JSON)
    collected_at = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, default=0.5)

class StyleProfile(Base):
    __tablename__ = 'style_profiles'

    id = Column(Integer, primary_key=True)
    artist_id = Column(Integer, ForeignKey('artists.id'), unique=True)
    text_description = Column(String)
    quantitative_params = Column(JSON)
    midi_templates_json = Column(JSON)  # List of file paths
    embedding = Column(Vector(384))  # sentence-transformers dimension
    confidence_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class GenerationHistory(Base):
    __tablename__ = 'generation_history'

    id = Column(Integer, primary_key=True)
    artist_id = Column(Integer, ForeignKey('artists.id'))
    provider_used = Column(String)  # openai/anthropic/google/template
    generation_time_ms = Column(Integer)
    user_params = Column(JSON)
    output_files = Column(JSON)  # List of generated MIDI paths
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Database Manager:**
```python
class DatabaseManager:
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)

    def get_or_create_artist(self, name: str) -> Artist:
        session = self.Session()
        artist = session.query(Artist).filter_by(name=name).first()
        if not artist:
            artist = Artist(name=name)
            session.add(artist)
            session.commit()
        return artist

    def get_style_profile(self, artist_name: str) -> Optional[StyleProfile]:
        session = self.Session()
        artist = session.query(Artist).filter_by(name=artist_name).first()
        if not artist:
            return None
        return session.query(StyleProfile).filter_by(artist_id=artist.id).first()

    def save_style_profile(self, profile: StyleProfile):
        session = self.Session()
        session.add(profile)
        session.commit()

    def find_similar_artists(
        self,
        embedding: np.ndarray,
        limit: int = 5
    ) -> List[Artist]:
        """Use pgvector for similarity search."""
        session = self.Session()
        results = session.query(
            StyleProfile.artist_id,
            StyleProfile.embedding.cosine_distance(embedding).label('distance')
        ).order_by('distance').limit(limit).all()

        artist_ids = [r.artist_id for r in results]
        return session.query(Artist).filter(Artist.id.in_(artist_ids)).all()
```

---

### SUB-AGENT 9: Ableton Integration Agent
**Role:** Max for Live device and Live API communication

**Implementation:** `ableton/MidiDrumGen.amxd` + `ableton/js/bridge.js`

**Max for Live Device Structure:**

**UI Components:**
1. Text input (`textedit`) - Artist name
2. Button (`live.button`) - Generate
3. Progress bar (`live.slider`) - Research/generation status
4. Number box (`live.numbox`) - Bars (1-16)
5. Number box - Tempo override
6. Number box - Variations count (1-8)
7. Button - "Augment Research"
8. Text display (`live.text`) - Status messages

**Max Patcher Logic:**
```
[textedit @parameter_enable 1 @varname artist_input]
    |
[live.button @parameter_enable 1 @varname generate_btn]
    |
[js bridge.js]  ← JavaScript bridge to FastAPI
    |
[live.path]  ← Import MIDI clips to Live
```

**JavaScript Bridge (`ableton/js/bridge.js`):**
```javascript
const API_BASE = "http://localhost:8000/api/v1";

async function generatePattern(artistName, params) {
    // Check if cached
    const cacheStatus = await fetch(`${API_BASE}/research/${artistName}`);
    const cached = await cacheStatus.json();

    if (!cached.exists) {
        // Start research
        updateStatus("Researching artist (this may take 5-20 minutes)...");
        const researchTask = await fetch(`${API_BASE}/research`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({artist: artistName})
        });

        const taskData = await researchTask.json();
        await pollTaskStatus(taskData.task_id);
    }

    // Generate MIDI
    updateStatus("Generating patterns...");
    const generateResp = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            artist: artistName,
            bars: params.bars,
            tempo: params.tempo,
            variations: params.variations
        })
    });

    const result = await generateResp.json();

    // Import MIDI files to Live
    for (const midiPath of result.midi_files) {
        importMidiClip(midiPath);
    }

    updateStatus(`✓ Generated ${result.midi_files.length} variations`);
}

function importMidiClip(midiPath) {
    // Use Live API to create clip
    const api = new LiveAPI("live_set");
    api.call("create_midi_clip_from_file", midiPath);
}

function pollTaskStatus(taskId) {
    return new Promise((resolve) => {
        const interval = setInterval(async () => {
            const resp = await fetch(`${API_BASE}/task/${taskId}`);
            const data = await resp.json();

            updateProgress(data.progress);

            if (data.status === 'completed') {
                clearInterval(interval);
                resolve();
            } else if (data.status === 'failed') {
                clearInterval(interval);
                updateStatus(`✗ Failed: ${data.error}`);
            }
        }, 2000);  // Poll every 2 seconds
    });
}
```

---

## Implementation Plan

### Phase 1: Infrastructure Setup (Week 1)

**Tasks:**
1. Setup PostgreSQL database with pgvector extension
2. Create Alembic migrations
3. Archive `src/training/` to `src/training_OLD/`
4. Create new directory structure for all agents
5. Setup environment variables (`.env` file)
6. Create base configuration files

**Directory Structure:**
```
src/
├── orchestrator/
│   └── main.py
├── research/
│   ├── orchestrator.py
│   ├── profile_builder.py
│   └── collectors/
│       ├── papers.py
│       ├── articles.py
│       ├── audio.py
│       └── midi_db.py
├── generation/
│   ├── providers/
│   │   ├── base.py
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── google.py
│   │   └── manager.py
│   ├── prompt_builder.py
│   ├── template_generator.py
│   └── hybrid_coordinator.py
├── database/
│   ├── models.py
│   ├── manager.py
│   └── migrations/
├── midi/
│   └── export_and_validate.py
└── ableton/
    ├── MidiDrumGen.amxd
    └── js/
        └── bridge.js
```

**Success Criteria:**
- Database schema created
- All directories exist
- Configuration files in place
- Dependencies installed

---

### Phase 2: Research Pipeline (Week 2-3)

**Week 2: Collectors**
- Implement Scholar Paper Collector
  - Test with 3 artists
  - Validate data extraction
- Implement Web Article Collector
  - Test scraping logic
  - Handle rate limits
- Implement Audio Analysis Collector
  - Test tempo/beat detection
  - Validate feature extraction
- Implement MIDI Database Collector
  - Test MIDI parsing
  - Validate pattern extraction

**Week 3: Orchestration**
- Build Research Orchestrator
  - Parallel execution of collectors
  - Error handling
  - Progress tracking
- Build Style Profile Builder
  - Data aggregation logic
  - Parameter extraction
  - Embedding generation
- Test with 10 diverse artists

**Success Criteria:**
- All collectors functional
- Research completes in 5-15 minutes
- StyleProfiles have >0.7 confidence
- Data stored in database correctly

---

### Phase 3: Generation Engine (Week 3-4)

**Week 3:**
- Implement OpenAI provider
- Implement Anthropic provider
- Implement Google provider
- Test each provider independently

**Week 4:**
- Build LLM Provider Manager with fallbacks
- Implement Prompt Engineering Agent
- Implement Template-Based Generator
- Build Hybrid Coordinator
- Test generation quality

**Success Criteria:**
- All 3 providers working
- Fallback logic works
- Generation completes < 2 minutes
- Output is valid MIDI
- Variations are diverse

---

### Phase 4: MIDI Export (Week 4)

**Tasks:**
- JSON to MIDI conversion
- Humanization pipeline
- Validation checks
- Batch export

**Success Criteria:**
- MIDI files import to DAWs correctly
- Humanization sounds natural
- No invalid MIDI generated

---

### Phase 5: Ableton Integration (Week 5)

**Tasks:**
- Build Max for Live UI
- Implement JavaScript bridge
- Test clip import to Live
- Handle error states in UI
- Add progress indicators

**Success Criteria:**
- Device loads in Ableton
- Artist input works
- Clips appear in clip slots
- Progress bar updates correctly

---

### Phase 6: End-to-End Testing (Week 6)

**Test Scenarios:**
1. Cached artist (< 2 min)
2. First-time artist (5-20 min)
3. Obscure artist (limited data)
4. Augmentation workflow
5. LLM provider failures
6. Multiple concurrent users

**Success Criteria:**
- All workflows complete successfully
- Performance targets met
- Error handling graceful
- UI responsive

---

## Technology Stack (Current Stable Versions - Nov 2025)

### Python Core
```
python>=3.11.0
```

### Web Framework & API
```
fastapi==0.109.2
uvicorn[standard]==0.27.1
celery==5.3.6
redis==5.0.1
```

### LLM Providers
```
openai==1.12.0
anthropic==0.18.1
google-generativeai==0.3.2
```

### Database
```
sqlalchemy==2.0.25
alembic==1.13.1
psycopg2-binary==2.9.9
pgvector==0.2.4
asyncpg==0.29.0
```

### Research & Web Scraping
```
beautifulsoup4==4.12.3
scrapy==2.11.1
newspaper3k==0.2.8
requests==2.31.0
aiohttp==3.9.3
```

### Audio Analysis
```
librosa==0.10.2.post1
madmom==0.16.1            # Advanced beat tracking (cross-platform, replaces Essentia)
yt-dlp==2024.12.13
soundfile==0.12.1
audioread==3.0.1
```

### NLP & Embeddings
```
spacy==3.7.4
sentence-transformers==2.3.1
numpy==1.26.4
```

### MIDI Processing
```
mido==1.3.2
python-rtmidi==1.5.8
```

### Utilities
```
python-dotenv==1.0.1
pydantic==2.6.1
pyyaml==6.0.1
loguru==0.7.2
```

---

## Configuration Files

### `configs/generation.yaml`
```yaml
llm:
  primary_provider: anthropic  # Claude 3.5 Sonnet
  fallback_providers: [anthropic, google]

  openai:
    model: gpt-4-turbo-preview
    api_key_env: OPENAI_API_KEY
    temperature: 0.8
    max_tokens: 2000

  anthropic:
    model: claude-3-opus-20240229
    api_key_env: ANTHROPIC_API_KEY
    temperature: 0.8
    max_tokens: 2000

  google:
    model: gemini-1.5-pro
    api_key_env: GOOGLE_API_KEY
    temperature: 0.8
    max_tokens: 2000

  local:
    enabled: false
    model: llama3-70b
    endpoint: http://localhost:11434

research:
  min_sources_per_type: 3
  timeout_minutes: 20
  quality_threshold: 0.7
  max_concurrent_collectors: 4

generation:
  default_bars: 4
  default_tempo: 120
  max_generation_time_sec: 120
  default_variations: 4
  humanization:
    micro_timing_ms: 10.0
    velocity_variation: 0.12

database:
  url_env: DATABASE_URL
  pool_size: 10
  max_overflow: 20

cache:
  enabled: true
  ttl_days: 30
```

### `.env`
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
DATABASE_URL=postgresql://user:pass@localhost:5432/mididrumigen_db
REDIS_URL=redis://localhost:6379/0
```

---

## Agent Communication Protocol

### Message Format
All inter-agent messages use standardized JSON format:

```python
{
    "command": str,  # Action to perform
    "payload": dict,  # Command-specific data
    "task_id": str,  # Unique task identifier
    "timestamp": str,  # ISO 8601
    "sender": str,  # Agent name
    "priority": int  # 1-10
}
```

### Orchestrator → Research Orchestrator
```json
{
    "command": "research_artist",
    "payload": {
        "artist_name": "John Bonham",
        "depth": "full"
    },
    "task_id": "uuid-123",
    "sender": "orchestrator"
}
```

### Research Orchestrator → Style Profile Builder
```json
{
    "command": "build_profile",
    "payload": {
        "artist_name": "John Bonham",
        "sources": {
            "papers": [...],
            "articles": [...],
            "audio_analysis": {...},
            "midi_templates": [...]
        }
    },
    "task_id": "uuid-123",
    "sender": "research_orchestrator"
}
```

### Orchestrator → LLM Manager
```json
{
    "command": "generate",
    "payload": {
        "artist_name": "John Bonham",
        "style_profile_id": 123,
        "params": {
            "bars": 4,
            "tempo": 120,
            "time_signature": [4, 4],
            "variations": 4
        },
        "provider": "openai"
    },
    "task_id": "uuid-456",
    "sender": "orchestrator"
}
```

---

## Success Criteria

### Functional Requirements
✓ User can input any artist name
✓ System researches if not cached (5-20 min)
✓ Generates MIDI in < 2 min for cached artists
✓ Generated patterns reflect artist's documented style
✓ Can augment research with more sources
✓ Clips appear directly in Ableton Live
✓ Works as Max for Live device

### Performance Requirements
✓ Database queries < 100ms
✓ LLM generation < 30s
✓ Total cached generation < 2 min
✓ Research pipeline < 20 min
✓ No memory leaks

### Quality Requirements
✓ Generated MIDI is valid and importable
✓ Style characteristics match research
✓ Multiple variations are diverse
✓ Humanization sounds natural
✓ Confidence score > 0.7 for usable profiles

---

## Implementation Instructions

**You are the Orchestrator Agent coordinating all sub-agents.**

**Your workflow:**

1. **Analyze existing codebase**
   - Review current training infrastructure
   - Identify reusable components
   - Archive old code

2. **Create directory structure**
   - Follow structure outlined above
   - Create placeholder files

3. **Implement Phase 1 (Infrastructure)**
   - Setup database
   - Create models
   - Test database operations

4. **Implement Phase 2 (Research Pipeline)**
   - Build collectors one by one
   - Test each collector independently
   - Integrate into orchestrator
   - Test with 10 artists

5. **Implement Phase 3 (Generation Engine)**
   - Implement LLM providers
   - Build prompt engineering
   - Create template generator
   - Build hybrid coordinator

6. **Implement Phase 4-6**
   - MIDI export
   - Ableton integration
   - End-to-end testing

**Testing at each phase is critical. Do not proceed until current phase passes tests.**

**Ask clarifying questions if any requirements are ambiguous.**

**Prioritize working code over perfect code - iterate and improve.**

---

## Notes

- **LLM Costs:** ~$0.01-0.05 per generation (acceptable)
- **Research Costs:** Mostly free APIs + web scraping
- **Audio Analysis:** Process locally (no API costs)
- **Database:** Self-hosted PostgreSQL
- **Max for Live:** Requires Max 8.5+ and Ableton Live 11+

---

**Ready to build! 🚀**
