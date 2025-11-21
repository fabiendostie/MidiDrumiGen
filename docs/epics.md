# Epics and User Stories
# MidiDrumiGen v2.0

**Project:** MidiDrumiGen v2.0 - On-Demand Artist-Style MIDI Generation
**Status:** Planning Phase
**Created:** 2025-11-18
**Last Updated:** 2025-11-18

---

## Epic Overview

This document breaks down the v2.0 PRD into deliverable epics and user stories. Each epic represents a major functional area that delivers value to end users. Stories within each epic are vertically sliced to deliver end-to-end functionality.

### Epic Summary

| Epic ID | Epic Name | User Value | Stories | Status |
|---------|-----------|------------|---------|--------|
| E1 | Research Pipeline | Automated artist style analysis | 8 | TODO |
| E2 | LLM Generation Engine | Multi-provider pattern generation | 7 | TODO |
| E3 | Database & Caching | Fast retrieval and style profiles | 6 | TODO |
| E4 | API Layer | RESTful service interface | 8 | TODO |
| E5 | MIDI Export & Humanization | Natural-sounding pattern export | 5 | TODO |
| E6 | Ableton Integration | Max for Live device | 6 | TODO |

**Total Stories:** 40

---

## Epic 1: Research Pipeline

**Goal:** Enable automated collection and analysis of artist drumming styles from multiple sources to create comprehensive StyleProfiles.

**User Value:** Users can get authentic drum patterns for any artist without manual research, powered by multi-source data collection.

**Dependencies:** None (foundational)

**Acceptance Criteria:**
- WHEN a user requests research for an artist
- THEN the system collects data from papers, articles, audio, and MIDI sources in parallel
- AND aggregates results into a StyleProfile with confidence score ≥ 0.6
- AND stores the profile in the database
- AND completes within 20 minutes (80th percentile)

---

### Story E1.S1: Scholar Paper Collection

**As a** system administrator
**I want** to search academic databases for papers analyzing artist drumming styles
**So that** StyleProfiles are backed by scholarly research

**Acceptance Criteria:**

```gherkin
Feature: Scholar Paper Collection
  Background:
    Given the Semantic Scholar API is accessible
    And the CrossRef API is accessible
    And the arXiv API is accessible

  Scenario: Successfully collect papers for well-documented artist
    Given an artist name "John Bonham"
    When the ScholarPaperCollector executes
    Then it searches Semantic Scholar with query "{artist} drumming rhythm analysis"
    And it searches arXiv with query "{artist} percussion music analysis"
    And it searches CrossRef with query "{artist} drum patterns"
    And it returns at least 3 ResearchSource objects of type 'paper'
    And each source has a confidence score ≥ 0.5
    And the operation completes within 5 minutes

  Scenario: Handle artist with limited academic coverage
    Given an artist name "Unknown Drummer"
    When the ScholarPaperCollector executes
    Then it searches all configured databases
    And it returns 0-2 ResearchSource objects
    And it logs a warning about limited coverage
    And it does not raise an exception

  Scenario: Extract tempo and style features from paper abstracts
    Given a paper abstract mentioning "typical tempo of 120 BPM"
    When the collector processes the abstract
    Then it extracts extracted_data with key 'tempo_mentions' containing [120]
    And it extracts style descriptors like 'syncopated', 'powerful', 'heavy'
    And it assigns confidence based on citation count
```

**Technical Notes:**
- File: `src/research/collectors/papers.py`
- Class: `ScholarPaperCollector(BaseCollector)`
- APIs: Semantic Scholar (100 req/5min), arXiv (unlimited), CrossRef (50 req/sec)
- Output: List[ResearchSource] with source_type='paper'

**Definition of Done:**
- [ ] ScholarPaperCollector class implemented
- [ ] Searches all 3 configured APIs
- [ ] Extracts tempo mentions using regex
- [ ] Assigns confidence scores based on citation count
- [ ] Unit tests for extraction logic pass
- [ ] Integration test with real API calls passes (slow test, marked)
- [ ] Handles rate limiting with exponential backoff
- [ ] Handles API failures gracefully (returns empty list, logs warning)
- [ ] Code reviewed and merged

---

### Story E1.S2: Web Article Collection

**As a** system administrator
**I want** to scrape music journalism sites for artist style descriptions
**So that** StyleProfiles include qualitative style characteristics

**Acceptance Criteria:**

```gherkin
Feature: Web Article Scraping
  Background:
    Given the following sites are accessible:
      | Site                  | URL Pattern                               |
      | Drummerworld          | drummerworld.com/drummers/{artist}        |
      | Wikipedia             | en.wikipedia.org/wiki/{artist}            |
      | Pitchfork             | pitchfork.com/search/?query={artist}      |
      | Rolling Stone         | rollingstone.com/search/{artist}          |
      | Sound On Sound        | soundonsound.com/search/all/{artist}      |
      | Music Connection      | musicconnection.com/?s={artist}           |
      | Music Business Worldwide | musicbusinessworldwide.com/?s={artist} |
      | The Pro Audio Files   | theproaudiofiles.com/?s={artist}          |
      | Produce Like a Pro    | producelikeapro.com/?s={artist}           |
      | Songstuff             | songstuff.com/?s={artist}                 |
      | Songwriter Universe   | songwriteruniverse.com/?s={artist}        |
      | Renegade Producer     | renegadeproducer.com/?s={artist}          |

  Scenario: Collect articles for well-documented artist
    Given an artist name "Travis Barker"
    When the WebArticleCollector executes
    Then it searches all configured sites
    And it extracts drumming-related content using NLP
    And it returns at least 5 ResearchSource objects of type 'article'
    And each article has raw_content with ≥ 3 drumming keyword mentions
    And the operation completes within 5 minutes

  Scenario: Filter non-drumming content using spaCy
    Given an article about "Travis Barker's fashion line"
    When the collector processes the article
    Then it counts keyword mentions (drum, beat, rhythm, etc.)
    And it finds < 3 keyword mentions
    And it excludes the article from results

  Scenario: Extract equipment and technique mentions
    Given an article mentioning "known for his fast kick drum technique"
    When the collector processes the article
    Then it extracts extracted_data with key 'techniques' containing ['fast kick drum']
    And it assigns confidence 0.6 (web article baseline)
```

**Technical Notes:**
- File: `src/research/collectors/articles.py`
- Class: `WebArticleCollector(BaseCollector)`
- Libraries: BeautifulSoup4, Scrapy, spaCy (en_core_web_sm)
- Output: List[ResearchSource] with source_type='article'

**Definition of Done:**
- [ ] WebArticleCollector class implemented
- [ ] Scrapes all 4 configured sites
- [ ] Uses spaCy NLP to filter drumming-related content
- [ ] Extracts equipment and technique mentions
- [ ] Handles HTTP errors (404, 503) gracefully
- [ ] Respects robots.txt
- [ ] Unit tests for NLP filtering pass
- [ ] Integration test with real scraping passes (slow test, marked)
- [ ] Code reviewed and merged

---

### Story E1.S3: Audio Analysis Collection

**As a** system administrator
**I want** to analyze audio recordings to extract rhythm features
**So that** StyleProfiles include quantitative tempo, swing, and velocity data

**Acceptance Criteria:**

```gherkin
Feature: Audio Analysis
  Background:
    Given yt-dlp is installed
    And Librosa is installed
    And madmom is installed

  Scenario: Analyze audio from YouTube
    Given an artist name "Questlove"
    When the AudioAnalysisCollector executes
    Then it searches YouTube for "{artist} drum solo" or "{artist} live performance"
    And it downloads up to 5 audio files using yt-dlp
    And it analyzes each file using Librosa and madmom
    And it extracts tempo_bpm, swing_ratio, syncopation_index, velocity distribution
    And it returns 3-5 ResearchSource objects of type 'audio'
    And each source has confidence ≥ 0.7
    And the operation completes within 8 minutes

  Scenario: Extract rhythm features using Librosa
    Given an audio file at 120 BPM with swing
    When the collector analyzes the file
    Then it detects tempo within ±5 BPM tolerance (115-125 BPM)
    And it calculates swing_ratio (beat timing ratio)
    And it extracts velocity_mean and velocity_std from RMS energy
    And it detects syncopation_index from off-beat accents

  Scenario: Use madmom for advanced beat tracking
    Given an audio file with complex rhythm
    When Librosa fails to detect beats accurately
    Then the collector falls back to madmom RNNBeatProcessor
    And it uses madmom TempoEstimationProcessor with confidence scores
    And it prefers madmom results over Librosa if confidence > 0.8

  Scenario: Handle audio download failures
    Given YouTube rate limits the download
    When the collector attempts to download audio
    Then it retries with exponential backoff (1s, 2s, 4s)
    And it skips the file after 3 failed attempts
    And it continues with remaining files
    And it logs a warning for the failed download
```

**Technical Notes:**
- File: `src/research/collectors/audio.py`
- Class: `AudioAnalysisCollector(BaseCollector)`
- Libraries: yt-dlp, Librosa 0.10.2.post1, madmom 0.16.1 (cross-platform)
- Output: List[ResearchSource] with source_type='audio'
- Madmom models: RNNBeatProcessor, TempoEstimationProcessor

**Definition of Done:**
- [ ] AudioAnalysisCollector class implemented
- [ ] Downloads audio using yt-dlp
- [ ] Analyzes with Librosa (primary) and madmom (advanced)
- [ ] Extracts tempo, swing, syncopation, velocities
- [ ] madmom integration for complex rhythms
- [ ] Cleans up temporary audio files after analysis
- [ ] Unit tests for feature extraction pass
- [ ] Integration test with real audio file passes (slow test)
- [ ] Handles download failures gracefully
- [ ] Code reviewed and merged

---

### Story E1.S4: MIDI Database Collection

**As a** system administrator
**I want** to search MIDI databases for existing artist patterns
**So that** StyleProfiles include authentic pattern templates

**Acceptance Criteria:**

```gherkin
Feature: MIDI Database Search
  Background:
    Given the following MIDI databases are accessible:
      | Database  | URL                                    |
      | BitMIDI   | bitmidi.com/search?q={artist}          |
      | FreeMIDI  | freemidi.org/search/{artist}           |
      | Musescore | musescore.com/sheetmusic?text={artist} |

  Scenario: Find and analyze MIDI files
    Given an artist name "Led Zeppelin"
    When the MidiDatabaseCollector executes
    Then it searches all 3 configured databases
    And it downloads up to 10 MIDI files
    And it extracts drum track (channel 10 or drum program)
    And it analyzes kick, snare, hihat patterns
    And it returns 2-4 ResearchSource objects of type 'midi'
    And each source has file_path to downloaded MIDI
    And the operation completes within 2 minutes

  Scenario: Extract drum patterns from MIDI track
    Given a MIDI file with drums on channel 10
    When the collector processes the file
    Then it identifies note 36 as kick drum
    And it identifies notes 38/40 as snare
    And it identifies notes 42/44/46 as hi-hat
    And it extracts patterns as {'kick': [times], 'snare': [times], 'hihat': [times]}
    And it stores patterns in extracted_data

  Scenario: Handle MIDI files without drum tracks
    Given a MIDI file with only piano and bass
    When the collector analyzes the file
    Then it finds no channel 10 messages
    And it skips the file
    And it continues with next file

  Scenario: Store MIDI templates for generation
    Given successful MIDI file analysis
    When the collector completes
    Then it saves MIDI files to data/midi_templates/{artist}/
    And it stores file paths in ResearchSource.file_path
    And it assigns confidence 0.9 (high for MIDI)
```

**Technical Notes:**
- File: `src/research/collectors/midi_db.py`
- Class: `MidiDatabaseCollector(BaseCollector)`
- Libraries: mido 1.3.3 (cross-platform)
- Output: List[ResearchSource] with source_type='midi'
- Storage: `data/midi_templates/{artist}/`

**Definition of Done:**
- [ ] MidiDatabaseCollector class implemented
- [ ] Searches all 3 configured databases
- [ ] Downloads MIDI files
- [ ] Extracts drum track (channel 10 detection)
- [ ] Analyzes kick/snare/hihat patterns
- [ ] Stores MIDI files in organized structure
- [ ] Unit tests for pattern extraction pass
- [ ] Integration test with real MIDI files passes
- [ ] Handles missing drum tracks gracefully
- [ ] Code reviewed and merged

---

### Story E1.S5: Style Profile Builder

**As a** system administrator
**I want** to aggregate research data into a unified StyleProfile
**So that** the generation engine has comprehensive artist characteristics

**Acceptance Criteria:**

```gherkin
Feature: Style Profile Aggregation
  Background:
    Given research results from all collectors:
      | Collector | Source Count |
      | Papers    | 5            |
      | Articles  | 8            |
      | Audio     | 3            |
      | MIDI      | 2            |

  Scenario: Build StyleProfile from multi-source data
    Given research data for "John Bonham"
    When the StyleProfileBuilder executes build()
    Then it extracts tempo from all sources
    And it calculates tempo_min, tempo_max, tempo_avg (filtering outliers)
    And it extracts swing from audio analysis
    And it calculates ghost_note_prob from qualitative descriptions
    And it generates text_description for LLM prompt
    And it creates vector embedding using sentence-transformers
    And it calculates confidence_score based on source count and quality
    And it returns a complete StyleProfile object

  Scenario: Resolve conflicting tempo values
    Given tempo values [120, 118, 125, 95, 122] from different sources
    When the builder aggregates tempo
    Then it filters outlier 95 (> 2 std dev from mean)
    And it calculates tempo_avg as mean of [118, 120, 122, 125] ≈ 121
    And it sets tempo_min = 118, tempo_max = 125

  Scenario: Generate textual description for LLM
    Given extracted style phrases ["powerful", "syncopated beats", "heavy bass drum"]
    When the builder generates text_description
    Then it combines phrases: "{artist} is known for powerful, syncopated beats and heavy bass drum"
    And it adds quantitative summary: "Typical tempo range: {min}-{max} BPM (average {avg} BPM)"
    And it adds swing info: "Swing feel at {swing}%" or "Straight timing"

  Scenario: Calculate confidence score
    Given sources: 5 papers, 8 articles, 3 audio, 2 MIDI (total 18)
    When the builder calculates confidence
    Then it assigns base confidence 0.5
    And it adds 0.02 per source (18 * 0.02 = 0.36)
    And it adds 0.1 bonus for MIDI presence
    And it caps at 1.0
    And confidence_score = min(1.0, 0.5 + 0.36 + 0.1) = 0.96

  Scenario: Create vector embedding for similarity search
    Given text_description for "J Dilla"
    When the builder creates embedding
    Then it uses SentenceTransformer('all-MiniLM-L6-v2')
    And it encodes the text_description
    And it returns a 384-dimensional vector
    And it stores in StyleProfile.embedding
```

**Technical Notes:**
- File: `src/research/profile_builder.py`
- Class: `StyleProfileBuilder`
- Libraries: sentence-transformers 2.3.1, numpy, scipy
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Output: StyleProfile dataclass

**Definition of Done:**
- [ ] StyleProfileBuilder class implemented
- [ ] Aggregates tempo with outlier filtering
- [ ] Extracts swing, ghost notes, syncopation from sources
- [ ] Generates textual description for LLM
- [ ] Creates 384-dim embedding using sentence-transformers
- [ ] Calculates confidence score based on source count/quality
- [ ] Unit tests for aggregation logic pass
- [ ] Integration test with mock research data passes
- [ ] Code reviewed and merged

---

### Story E1.S6: Research Orchestrator

**As a** system administrator
**I want** to coordinate all research collectors in parallel
**So that** research completes within 20 minutes

**Acceptance Criteria:**

```gherkin
Feature: Research Orchestration
  Background:
    Given all 4 collectors are initialized
    And the StyleProfileBuilder is initialized
    And the database connection is established

  Scenario: Coordinate parallel research
    Given a research request for "Travis Barker"
    When the ResearchOrchestrator executes research_artist()
    Then it runs all 4 collectors in parallel using asyncio.gather()
    And it waits for all to complete or timeout (20 min)
    And it aggregates results using StyleProfileBuilder
    And it stores StyleProfile in database
    And it updates Artist record with research_status='cached'
    And it returns the StyleProfile
    And total time < 20 minutes (excluding timeout cases)

  Scenario: Handle partial collector failures
    Given the audio collector times out
    And the other 3 collectors succeed
    When the orchestrator aggregates results
    Then it includes data from papers, articles, MIDI
    And it logs a warning about audio failure
    And it proceeds with StyleProfile building
    And confidence_score is reduced due to missing audio data

  Scenario: Report progress to caller
    Given a progress_callback function
    When the orchestrator executes research_artist()
    Then it calls progress_callback(0, "Starting research...")
    And it calls progress_callback(20, "Collecting papers...")
    And it calls progress_callback(40, "Scraping articles...")
    And it calls progress_callback(60, "Analyzing audio...")
    And it calls progress_callback(75, "Searching MIDI...")
    And it calls progress_callback(90, "Building profile...")
    And it calls progress_callback(100, "Research complete!")

  Scenario: Validate minimum quality standards
    Given only 3 total sources found
    And the configured minimum is 8 sources
    When the orchestrator validates the profile
    Then it raises InsufficientDataError
    And it does not store the profile in database
    And it sets Artist.research_status = 'failed'

  Scenario: Store research metadata
    Given successful research completion
    When the orchestrator stores the profile
    Then it creates Artist record with:
      | Field              | Value              |
      | name               | Travis Barker      |
      | research_status    | cached             |
      | sources_count      | 18                 |
      | confidence_score   | 0.85               |
      | last_updated       | current timestamp  |
    And it creates StyleProfile record linked to Artist
    And it creates 18 ResearchSource records
```

**Technical Notes:**
- File: `src/research/orchestrator.py`
- Class: `ResearchOrchestrator`
- Pattern: Async orchestration with asyncio.gather()
- Timeout: 20 minutes default
- Quality threshold: ≥ 8 total sources, confidence ≥ 0.6

**Definition of Done:**
- [ ] ResearchOrchestrator class implemented
- [ ] Runs collectors in parallel using asyncio.gather()
- [ ] Handles timeout with partial results
- [ ] Reports progress via callback
- [ ] Validates minimum quality standards
- [ ] Stores StyleProfile and metadata in database
- [ ] Handles collector failures gracefully
- [ ] Unit tests for orchestration logic pass
- [ ] Integration test with all collectors passes (very slow test)
- [ ] Code reviewed and merged

---

### Story E1.S7: Research API Endpoint

**As a** API consumer
**I want** to trigger artist research via REST endpoint
**So that** I can initiate research from Max for Live or CLI

**Acceptance Criteria:**

```gherkin
Feature: Research API
  Background:
    Given the FastAPI server is running
    And the ResearchOrchestrator is configured

  Scenario: Trigger research for new artist
    Given the artist "Elvin Jones" is not in the database
    When I POST to /api/v1/research with {"artist": "Elvin Jones"}
    Then the response status is 202 Accepted
    And the response body contains:
      | Field                   | Value                                |
      | task_id                 | <UUID>                               |
      | status                  | researching                          |
      | estimated_time_minutes  | 15                                   |
    And a Celery task is enqueued to research_queue

  Scenario: Check research status
    Given a research task with task_id "abc-123"
    When I GET /api/v1/task/abc-123
    Then the response status is 200 OK
    And the response body contains:
      | Field         | Value                           |
      | status        | in_progress                     |
      | progress      | 45                              |
      | current_step  | Analyzing audio...              |

  Scenario: Research completion
    Given a completed research task
    When I GET /api/v1/task/abc-123
    Then the response status is 200 OK
    And the response body contains:
      | Field         | Value          |
      | status        | completed      |
      | progress      | 100            |
      | result        | {profile data} |

  Scenario: Check if artist is cached
    Given the artist "Travis Barker" is researched
    When I GET /api/v1/research/Travis%20Barker
    Then the response status is 200 OK
    And the response body contains:
      | Field         | Value                    |
      | exists        | true                     |
      | confidence    | 0.85                     |
      | last_updated  | 2025-11-15T10:30:00Z     |
      | sources_count | {"papers": 5, "articles": 12, "audio": 3, "midi": 2} |
```

**Technical Notes:**
- File: `src/api/routes/research.py`
- Endpoints: POST /api/v1/research, GET /api/v1/research/{artist}, GET /api/v1/task/{task_id}
- Task queue: Celery with Redis broker
- Queue name: 'research'

**Definition of Done:**
- [ ] POST /api/v1/research endpoint implemented
- [ ] GET /api/v1/research/{artist} endpoint implemented
- [ ] GET /api/v1/task/{task_id} endpoint implemented
- [ ] Celery task for research created
- [ ] Returns 202 Accepted with task_id
- [ ] Task status polling works
- [ ] Unit tests for endpoints pass
- [ ] Integration test with Celery worker passes
- [ ] API documentation updated (OpenAPI/Swagger)
- [ ] Code reviewed and merged

---

### Story E1.S8: Augmentation Feature

**As a** user
**I want** to add more sources to an existing StyleProfile
**So that** I can improve generation quality over time

**Acceptance Criteria:**

```gherkin
Feature: Style Profile Augmentation
  Background:
    Given an artist "J Dilla" with existing StyleProfile
    And the profile has confidence_score 0.75
    And the profile has 12 total sources

  Scenario: Augment existing profile
    Given the user requests augmentation for "J Dilla"
    When I POST to /api/v1/augment/J%20Dilla
    Then the response status is 202 Accepted
    And the response body contains:
      | Field           | Value           |
      | task_id         | <UUID>          |
      | status          | augmenting      |
      | current_sources | 12              |
      | target_sources  | 17-20           |
    And the collectors run again to find additional sources

  Scenario: Merge new sources with existing
    Given the augmentation finds 5 new sources
    When the orchestrator rebuilds the profile
    Then it merges new sources with existing 12 sources
    And it rebuilds StyleProfile with 17 total sources
    And confidence_score increases from 0.75 to 0.85
    And it updates the StyleProfile in database

  Scenario: Improve generation quality after augmentation
    Given generation quality before augmentation rated 3/5
    And augmentation adds 5 high-quality audio sources
    When generation occurs after augmentation
    Then the LLM has more detailed style information
    And generated patterns better match artist style
    And user survey ratings improve to 4/5
```

**Technical Notes:**
- File: `src/research/orchestrator.py` (new method `augment_artist()`)
- Endpoint: POST /api/v1/augment/{artist}
- Strategy: Run collectors again, merge with existing sources

**Definition of Done:**
- [ ] ResearchOrchestrator.augment_artist() method implemented
- [ ] POST /api/v1/augment/{artist} endpoint implemented
- [ ] Merges new sources with existing
- [ ] Updates confidence score
- [ ] Updates StyleProfile in database
- [ ] Unit tests for merge logic pass
- [ ] Integration test for augmentation passes
- [ ] API documentation updated
- [ ] Code reviewed and merged

---

## Epic 2: LLM Generation Engine

**Goal:** Enable multi-provider LLM-based MIDI pattern generation with automatic failover and quality assurance.

**User Value:** Users get authentic, high-quality drum patterns generated by state-of-the-art LLMs with 98%+ success rate through provider fallback.

**Dependencies:** Epic 1 (StyleProfile must exist)

**Acceptance Criteria:**
- WHEN a user requests pattern generation for a cached artist
- THEN the system generates MIDI using Claude 4.5 Sonnet (primary), Gemini 2.5 Pro (secondary), or ChatGPT 5.1 (fallback)
- AND validates the output structure
- AND applies style-specific humanization
- AND returns 4-8 MIDI file paths
- AND completes within 2 minutes

---

### Story E2.S1: LLM Provider Interface

**As a** developer
**I want** a unified interface for all LLM providers
**So that** the system can swap providers seamlessly

**Acceptance Criteria:**

```gherkin
Feature: LLM Provider Interface
  Scenario: Define base provider interface
    Given the BaseLLMProvider abstract class
    Then it defines generate_midi_tokens(prompt, system_prompt) -> dict
    And it defines validate_credentials() -> bool
    And it defines provider_name property
    And it defines cost_per_1k_tokens property
    And it implements record_success(tokens, cost) method
    And it implements record_failure() method
    And it tracks total_requests, total_failures, tokens_used, cost_usd

  Scenario: Track provider performance
    Given a provider has processed 100 requests
    And 5 requests failed
    When I access provider.success_rate
    Then it returns 0.95 (95%)

  Scenario: Calculate generation cost
    Given a generation using 2000 tokens
    And the provider costs $0.01 input, $0.03 output per 1K tokens
    When I calculate cost
    Then input_cost = 1000 tokens * $0.01/1K = $0.01
    And output_cost = 1000 tokens * $0.03/1K = $0.03
    And total_cost = $0.04
```

**Technical Notes:**
- File: `src/generation/providers/base.py`
- Class: `BaseLLMProvider` (ABC)
- Metrics: requests, failures, tokens, cost

**Definition of Done:**
- [ ] BaseLLMProvider abstract class implemented
- [ ] All abstract methods defined
- [ ] Performance tracking implemented
- [ ] Cost calculation implemented
- [ ] Unit tests for tracking logic pass
- [ ] Code reviewed and merged

---

### Story E2.S2: Claude Provider Implementation

**As a** developer
**I want** to implement Anthropic Claude 4.5 Sonnet provider
**So that** the system can use Claude as primary LLM

**Acceptance Criteria:**

```gherkin
Feature: Claude Provider
  Background:
    Given ANTHROPIC_API_KEY is set in environment
    And Anthropic Python SDK is installed

  Scenario: Generate MIDI using Claude 4.5 Sonnet
    Given a prompt with artist style and parameters
    And a system prompt with format specifications
    When the AnthropicProvider calls generate_midi_tokens()
    Then it creates AsyncAnthropic client
    And it calls messages.create() with model "claude-4-5-sonnet-20250514"
    And it sets temperature=0.8, max_tokens=4096
    And it parses the response as JSON
    And it returns {"notes": [...], "tempo": 120, "time_signature": [4,4]}

  Scenario: Handle Claude wrapping JSON in markdown
    Given Claude returns "```json\n{...}\n```"
    When the provider parses the response
    Then it extracts JSON using regex r'```json\n(.*?)\n```'
    And it returns the parsed JSON object

  Scenario: Calculate Claude cost
    Given a generation using 2500 tokens
    And Claude Sonnet costs $0.003 input, $0.015 output per 1K
    When the provider calculates cost
    Then input_cost ≈ $0.00375 (1250 tokens)
    And output_cost ≈ $0.01875 (1250 tokens)
    And total_cost ≈ $0.0225
```

**Technical Notes:**
- File: `src/generation/providers/anthropic_provider.py`
- Class: `AnthropicProvider(BaseLLMProvider)`
- Model: claude-4-5-sonnet-20250514
- Cost: $3/$15 per 1M tokens (input/output)

**Definition of Done:**
- [ ] AnthropicProvider class implemented
- [ ] Inherits from BaseLLMProvider
- [ ] Implements generate_midi_tokens() using Anthropic SDK
- [ ] Handles markdown-wrapped JSON
- [ ] Implements validate_credentials()
- [ ] Returns cost_per_1k_tokens
- [ ] Unit tests with mocked API pass
- [ ] Integration test with real API passes (slow test, requires key)
- [ ] Code reviewed and merged

---

### Story E2.S3: Gemini Provider Implementation

**As a** developer
**I want** to implement Google Gemini 2.5 Pro provider
**So that** the system has a fast, low-cost secondary LLM

**Acceptance Criteria:**

```gherkin
Feature: Gemini Provider
  Background:
    Given GOOGLE_API_KEY is set in environment
    And google-generativeai SDK is installed

  Scenario: Generate MIDI using Gemini 2.5 Pro
    Given a prompt with artist style and parameters
    And a system prompt with format specifications
    When the GoogleProvider calls generate_midi_tokens()
    Then it combines system_prompt + user_prompt (Gemini doesn't have separate system)
    And it creates GenerativeModel('gemini-2.5-flash') (or 'gemini-3.0-pro' when available)
    And it calls generate_content_async() with temperature=0.8
    And it parses response.text as JSON
    And it returns {"notes": [...], "tempo": 120, "time_signature": [4,4]}

  Scenario: Calculate Gemini cost
    Given a generation using 2000 tokens
    And Gemini 2.5 Pro costs $0.000125 input, $0.000375 output per 1K
    When the provider calculates cost
    Then input_cost ≈ $0.000125
    And output_cost ≈ $0.000375
    And total_cost ≈ $0.0005 (very cheap)
```

**Technical Notes:**
- File: `src/generation/providers/google_provider.py`
- Class: `GoogleProvider(BaseLLMProvider)`
- Model: gemini-2.5-flash (preferred) or gemini-3.0-pro
- Cost: $0.125/$0.375 per 1M tokens (Flash)

**Definition of Done:**
- [ ] GoogleProvider class implemented
- [ ] Inherits from BaseLLMProvider
- [ ] Implements generate_midi_tokens() using Google SDK
- [ ] Combines system + user prompts
- [ ] Implements validate_credentials()
- [ ] Returns cost_per_1k_tokens
- [ ] Unit tests with mocked API pass
- [ ] Integration test with real API passes
- [ ] Code reviewed and merged

---

### Story E2.S4: OpenAI Provider Implementation

**As a** developer
**I want** to implement OpenAI ChatGPT 5.1 provider
**So that** the system has a reliable fallback LLM

**Acceptance Criteria:**

```gherkin
Feature: OpenAI Provider
  Background:
    Given OPENAI_API_KEY is set in environment
    And openai Python SDK is installed

  Scenario: Generate MIDI using ChatGPT 5.1
    Given a prompt with artist style and parameters
    And a system prompt with format specifications
    When the OpenAIProvider calls generate_midi_tokens()
    Then it creates AsyncOpenAI client
    And it calls chat.completions.create() with model "chatgpt-5.1-latest"
    And it sets response_format={"type": "json_object"} to force JSON
    And it sets temperature=0.8, max_tokens=4096
    And it parses response.choices[0].message.content
    And it returns {"notes": [...], "tempo": 120, "time_signature": [4,4]}

  Scenario: Force JSON output format
    Given response_format={"type": "json_object"}
    When OpenAI generates response
    Then it guarantees valid JSON output
    And no markdown wrapping is needed

  Scenario: Calculate OpenAI cost
    Given a generation using 2000 tokens
    And ChatGPT 5.1 costs $0.0025 input, $0.01 output per 1K
    When the provider calculates cost
    Then input_cost ≈ $0.0025
    And output_cost ≈ $0.01
    And total_cost ≈ $0.0125
```

**Technical Notes:**
- File: `src/generation/providers/openai_provider.py`
- Class: `OpenAIProvider(BaseLLMProvider)`
- Model: chatgpt-5.1-latest (ChatGPT 5.1)
- Cost: $2.50/$10 per 1M tokens

**Definition of Done:**
- [ ] OpenAIProvider class implemented
- [ ] Inherits from BaseLLMProvider
- [ ] Implements generate_midi_tokens() using OpenAI SDK
- [ ] Forces JSON output format
- [ ] Implements validate_credentials()
- [ ] Returns cost_per_1k_tokens
- [ ] Unit tests with mocked API pass
- [ ] Integration test with real API passes
- [ ] Code reviewed and merged

---

### Story E2.S5: LLM Provider Manager with Failover

**As a** system
**I want** to automatically fallback to secondary LLMs when primary fails
**So that** generation has 98%+ success rate

**Acceptance Criteria:**

```gherkin
Feature: LLM Provider Manager
  Background:
    Given primary_provider is "anthropic"
    And fallback_providers are ["google", "openai"]
    And all providers are initialized

  Scenario: Successful generation with primary provider
    Given a generation request
    When the LLMProviderManager calls generate()
    Then it attempts Claude 4.5 Sonnet first
    And Claude succeeds
    And it returns:
      | Field           | Value                                  |
      | result          | {"notes": [...]}                       |
      | provider_used   | anthropic                              |
      | tokens_used     | 1847                                   |
      | cost_usd        | 0.0123                                 |
    And it records success for Claude provider

  Scenario: Fallback to secondary provider on primary failure
    Given Claude API returns 503 Service Unavailable
    When the manager attempts generation
    Then it logs "anthropic failed: 503 error"
    And it tries Gemini 2.5 Pro next
    And Gemini succeeds
    And it returns:
      | Field           | Value                                  |
      | provider_used   | google                                 |
    And it records failure for Claude
    And it records success for Gemini

  Scenario: All providers fail
    Given all 3 providers return errors
    When the manager attempts generation
    Then it tries Claude → fails
    And it tries Gemini → fails
    And it tries OpenAI → fails
    And it raises LLMProviderError("All providers failed. Last error: ...")
    And it records failures for all providers

  Scenario: Track provider statistics
    Given the manager has processed 100 requests
    And Claude succeeded 80 times, failed 5 times
    And Gemini succeeded 10 times (fallback)
    And OpenAI succeeded 5 times (fallback)
    When I call get_provider_stats()
    Then it returns:
      | Provider   | Requests | Failures | Success Rate | Cost     |
      | anthropic  | 85       | 5        | 94.1%        | $1.85    |
      | google     | 10       | 0        | 100%         | $0.05    |
      | openai     | 5        | 0        | 100%         | $0.08    |
```

**Technical Notes:**
- File: `src/generation/providers/manager.py`
- Class: `LLMProviderManager`
- Fallback order: Claude → Gemini → OpenAI
- Metrics: per-provider success rate, cost, tokens

**Definition of Done:**
- [ ] LLMProviderManager class implemented
- [ ] Initializes all configured providers
- [ ] Implements generate() with failover logic
- [ ] Tries primary provider first
- [ ] Falls back to secondary/tertiary on failure
- [ ] Tracks provider statistics
- [ ] Returns generation metadata (provider, tokens, cost)
- [ ] Unit tests with mocked providers pass
- [ ] Integration test with failover scenario passes
- [ ] Code reviewed and merged

---

### Story E2.S6: Prompt Engineering

**As a** developer
**I want** to build optimized prompts for drum pattern generation
**So that** LLMs produce high-quality, valid MIDI

**Acceptance Criteria:**

```gherkin
Feature: Prompt Building
  Scenario: Build system prompt with format specification
    Given the SYSTEM_PROMPT constant
    Then it defines the LLM's role as "expert AI drummer"
    And it specifies JSON output format with notes array
    And it defines MIDI note mappings (36=Kick, 38=Snare, etc.)
    And it lists rules:
      | Rule                                    |
      | Output ONLY valid JSON                  |
      | Time values align to grid (multiples of 30) |
      | Velocity range 1-127                    |
      | Duration minimum 30 ticks               |
      | Pattern must be musically coherent      |
      | Include realistic velocity variation    |
      | Follow artist's style characteristics   |

  Scenario: Build user prompt with StyleProfile
    Given a StyleProfile for "J Dilla"
    And generation parameters: 4 bars, 95 BPM, 4/4 time
    When I call build_user_prompt(profile, params)
    Then it includes artist name
    And it includes text_description from profile
    And it includes quantitative parameters:
      | Parameter                | Value                     |
      | Tempo                    | 95 BPM (typical 85-120)   |
      | Swing                    | 62%                       |
      | Ghost note probability   | 0.4                       |
      | Average velocity         | 85                        |
    And it includes MIDI template examples (if available)
    And it ends with "Output JSON only, no explanation"

  Scenario: Include MIDI template examples
    Given the profile has 2 MIDI templates
    When building the prompt
    Then it extracts first 8 notes from each template
    And it formats as JSON examples
    And it includes in prompt as "SIGNATURE PATTERNS (reference):"
```

**Technical Notes:**
- File: `src/generation/prompt_builder.py`
- Functions: `build_user_prompt(profile, params)`, format_midi_templates()
- Constants: SYSTEM_PROMPT

**Definition of Done:**
- [ ] SYSTEM_PROMPT constant defined
- [ ] build_user_prompt() function implemented
- [ ] Includes style characteristics
- [ ] Includes quantitative parameters
- [ ] Includes MIDI template examples
- [ ] format_midi_templates() helper implemented
- [ ] Unit tests for prompt building pass
- [ ] Prompt tested with all 3 LLM providers
- [ ] Code reviewed and merged

---

### Story E2.S7: Hybrid Generation Coordinator

**As a** system
**I want** to coordinate LLM and template-based generation
**So that** the system has 100% success rate with fallback

**Acceptance Criteria:**

```gherkin
Feature: Hybrid Generation
  Background:
    Given the LLMProviderManager is initialized
    And the TemplateGenerator is initialized
    And a StyleProfile for "John Bonham" exists

  Scenario: Generate using LLM successfully
    Given a generation request for "John Bonham"
    When the HybridCoordinator calls generate()
    Then it builds user prompt with StyleProfile
    And it calls LLMProviderManager.generate()
    And LLM returns valid MIDI JSON
    And it validates the output structure
    And validation passes
    And it applies style transfer
    And it generates 4 variations
    And it exports to MIDI files
    And it returns list of file paths

  Scenario: Fallback to template generator on LLM failure
    Given all LLM providers fail
    When the coordinator attempts generation
    Then it logs "LLM generation failed, using templates"
    And it calls TemplateGenerator.generate()
    And it loads MIDI templates from StyleProfile
    And it applies style variations (swing, ghost notes, velocities)
    And it converts to JSON format
    And it returns valid MIDI data

  Scenario: Validate LLM output structure
    Given LLM returns {"notes": [...], "tempo": 120, "time_signature": [4,4]}
    When the coordinator validates output
    Then it checks for required keys: notes, tempo, time_signature
    And it validates each note has: pitch, velocity, time, duration
    And it checks pitch ranges (35-81 for drums)
    And it checks velocity ranges (1-127)
    And validation passes

  Scenario: Create variations from base pattern
    Given a base pattern with 32 notes
    And request for 4 variations
    When the coordinator creates variations
    Then it keeps variation 1 as original
    And it creates variation 2 with randomized velocities (±10%)
    And it creates variation 3 with timing shifts (±15 ticks)
    And it creates variation 4 with added ghost notes
    And it returns 4 distinct MIDI JSON objects
```

**Technical Notes:**
- File: `src/generation/hybrid_coordinator.py`
- Class: `HybridCoordinator`
- Dependencies: LLMProviderManager, TemplateGenerator
- Fallback: LLM → Templates → Error

**Definition of Done:**
- [ ] HybridCoordinator class implemented
- [ ] Tries LLM generation first
- [ ] Validates LLM output structure
- [ ] Falls back to TemplateGenerator on failure
- [ ] Creates variations from base pattern
- [ ] Applies style transfer
- [ ] Unit tests for validation logic pass
- [ ] Integration test with both paths (LLM + template) passes
- [ ] Code reviewed and merged

---

## Epic 3: Database & Caching

**Goal:** Enable fast retrieval of StyleProfiles using PostgreSQL + pgvector with similarity search.

**User Value:** Users get near-instant pattern generation for cached artists (< 2 min) and can discover similar artists.

**Dependencies:** Phase 2 infrastructure (PostgreSQL + pgvector setup complete)

**Acceptance Criteria:**
- WHEN a user requests a cached artist
- THEN the system retrieves StyleProfile from database in < 100ms
- AND supports vector similarity search for "artists like X"
- AND tracks all generation history for analytics

---

### Story E3.S1: Database Models

**As a** developer
**I want** to define SQLAlchemy models for all entities
**So that** the system can persist data in PostgreSQL

**Acceptance Criteria:**

```gherkin
Feature: Database Models
  Scenario: Define Artist model
    Given the Artist model
    Then it has columns:
      | Column           | Type        | Constraints        |
      | id               | UUID        | Primary Key        |
      | name             | String(255) | Unique, Not Null   |
      | research_status  | String(50)  | Default 'pending'  |
      | last_updated     | DateTime    | Auto-update        |
      | sources_count    | Integer     | Default 0          |
      | confidence_score | Float       | Default 0.0        |
      | created_at       | DateTime    | Auto-set           |
    And it has index on name
    And it has relationship to StyleProfile (one-to-one)
    And it has relationship to ResearchSources (one-to-many)

  Scenario: Define StyleProfile model
    Given the StyleProfile model
    Then it has columns:
      | Column               | Type       | Constraints                        |
      | id                   | UUID       | Primary Key                        |
      | artist_id            | UUID       | Foreign Key (artists), Unique      |
      | text_description     | Text       | Not Null                           |
      | quantitative_params  | JSON       | Not Null                           |
      | midi_templates_json  | JSON       | Nullable                           |
      | embedding            | Vector(384)| Nullable (pgvector)                |
      | confidence_score     | Float      | Not Null                           |
      | sources_count        | JSON       | Nullable                           |
      | created_at           | DateTime   | Auto-set                           |
      | updated_at           | DateTime   | Auto-update                        |
    And it has vector index on embedding (IVFFlat, cosine distance)
    And it has unique constraint on artist_id

  Scenario: Define ResearchSource model
    Given the ResearchSource model
    Then it has columns:
      | Column         | Type      | Constraints              |
      | id             | UUID      | Primary Key              |
      | artist_id      | UUID      | Foreign Key (artists)    |
      | source_type    | String(50)| Not Null                 |
      | url            | Text      | Nullable                 |
      | file_path      | Text      | Nullable                 |
      | raw_content    | Text      | Nullable                 |
      | extracted_data | JSON      | Nullable                 |
      | confidence     | Float     | Default 0.5              |
      | collected_at   | DateTime  | Auto-set                 |
    And it has composite index on (artist_id, source_type)

  Scenario: Define GenerationHistory model
    Given the GenerationHistory model
    Then it has columns:
      | Column              | Type       | Constraints           |
      | id                  | UUID       | Primary Key           |
      | artist_id           | UUID       | Foreign Key (artists) |
      | provider_used       | String(50) | Not Null              |
      | generation_time_ms  | Integer    | Not Null              |
      | tokens_used         | Integer    | Nullable              |
      | cost_usd            | Float      | Nullable              |
      | user_params         | JSON       | Not Null              |
      | output_files        | JSON       | Nullable              |
      | created_at          | DateTime   | Auto-set              |
    And it has index on artist_id
    And it has index on created_at
```

**Technical Notes:**
- File: `src/database/models.py`
- ORM: SQLAlchemy 2.0+
- UUID: Using uuid.uuid4() as default
- Vector: pgvector.sqlalchemy.Vector(384)

**Definition of Done:**
- [ ] All 4 models defined in src/database/models.py
- [ ] All columns and constraints specified
- [ ] Relationships defined (one-to-one, one-to-many)
- [ ] Indexes created (name, artist_id, created_at, embedding)
- [ ] Vector index configured (IVFFlat, cosine)
- [ ] Base metadata exported
- [ ] Code reviewed and merged

---

### Story E3.S2: Database Manager

**As a** developer
**I want** a database manager for CRUD operations
**So that** I don't write raw SQL queries

**Acceptance Criteria:**

```gherkin
Feature: Database Manager
  Scenario: Save StyleProfile
    Given a StyleProfile object for "J Dilla"
    When I call db.save_style_profile(profile)
    Then it creates or updates Artist record
    And it creates or updates StyleProfile record
    And it commits the transaction
    And it returns the saved profile

  Scenario: Get StyleProfile by artist name
    Given "Travis Barker" exists in database
    When I call db.get_style_profile("Travis Barker")
    Then it queries StyleProfile joined with Artist
    And it returns the StyleProfile object
    And the query completes in < 100ms

  Scenario: Search similar artists using vector similarity
    Given a StyleProfile for "J Dilla" with embedding
    When I call db.find_similar_artists("J Dilla", limit=5)
    Then it queries StyleProfiles ordered by cosine distance
    And it returns 5 most similar artists (excluding J Dilla)
    And the query uses the pgvector index
    And the query completes in < 200ms

  Scenario: Save generation history
    Given a completed generation
    When I call db.save_generation_history(history)
    Then it creates GenerationHistory record
    And it stores provider_used, tokens, cost, params
    And it commits the transaction

  Scenario: Get artist by name (create if not exists)
    Given the artist "New Artist" doesn't exist
    When I call db.get_or_create_artist("New Artist")
    Then it creates new Artist record
    And it sets research_status = 'pending'
    And it returns the Artist object
```

**Technical Notes:**
- File: `src/database/manager.py`
- Class: `DatabaseManager`
- Session: Async session management
- Connection pool: SQLAlchemy engine with pooling

**Definition of Done:**
- [ ] DatabaseManager class implemented
- [ ] save_style_profile() method implemented
- [ ] get_style_profile() method implemented
- [ ] find_similar_artists() method implemented (vector search)
- [ ] save_generation_history() method implemented
- [ ] get_or_create_artist() method implemented
- [ ] Async session context manager
- [ ] Unit tests with in-memory database pass
- [ ] Integration tests with PostgreSQL pass
- [ ] Code reviewed and merged

---

### Story E3.S3: Alembic Migrations

**As a** developer
**I want** to manage schema changes with Alembic
**So that** database migrations are versioned and reversible

**Acceptance Criteria:**

```gherkin
Feature: Database Migrations
  Scenario: Initial schema migration exists
    Given the alembic/versions/ directory
    Then it contains one migration file: a86af6571aac_initial_v2_0_schema.py
    And the migration creates tables: artists, style_profiles, research_sources, generation_history
    And it creates pgvector extension
    And it creates vector index on style_profiles.embedding

  Scenario: Apply migrations to empty database
    Given a fresh PostgreSQL database
    When I run `alembic upgrade head`
    Then it creates all 4 tables
    And it creates pgvector extension
    And it creates all indexes
    And alembic_version table shows current revision

  Scenario: Rollback migration
    Given migrations are applied
    When I run `alembic downgrade -1`
    Then it drops all tables
    And it drops pgvector extension
    And database is reverted to previous state
```

**Technical Notes:**
- Tool: Alembic 1.13+
- Config: alembic.ini, alembic/env.py
- Migrations: alembic/versions/*.py
- Current migration: a86af6571aac (already created in Phase 2)

**Definition of Done:**
- [ ] Alembic configured and working (already done in Phase 2)
- [ ] Initial migration creates all tables
- [ ] Migration creates pgvector extension
- [ ] Migration creates vector index
- [ ] `alembic upgrade head` works
- [ ] `alembic downgrade` works
- [ ] Documentation updated with migration commands
- [ ] Verified in Phase 2 completion

---

### Story E3.S4: Vector Similarity Search

**As a** user
**I want** to find artists similar to a given artist
**So that** I can discover new drumming styles

**Acceptance Criteria:**

```gherkin
Feature: Vector Similarity Search
  Background:
    Given StyleProfiles exist for:
      | Artist         | Tempo | Swing | Genre   |
      | J Dilla        | 95    | 62%   | Hip-Hop |
      | Questlove      | 90    | 58%   | Neo-Soul|
      | Travis Barker  | 180   | 0%    | Punk    |
      | John Bonham    | 110   | 55%   | Rock    |

  Scenario: Find similar artists to J Dilla
    Given the user queries similar artists to "J Dilla"
    When I call db.find_similar_artists("J Dilla", limit=3)
    Then it returns:
      | Rank | Artist      | Similarity |
      | 1    | Questlove   | 0.91       |
      | 2    | John Bonham | 0.73       |
      | 3    | Travis Barker| 0.42      |
    And Questlove is most similar (similar tempo, swing, genre)
    And Travis Barker is least similar (very different tempo/style)

  Scenario: API endpoint for similarity search
    Given the GET /api/v1/similar/{artist} endpoint
    When I GET /api/v1/similar/J%20Dilla?limit=3
    Then the response status is 200 OK
    And the response body contains:
      | Field           | Value                          |
      | artist          | J Dilla                        |
      | similar_artists | [{"name": "Questlove", "similarity": 0.91}, ...] |

  Scenario: Optimize vector search performance
    Given 10,000 StyleProfiles in database
    When I query similar artists
    Then the query uses IVFFlat index
    And the query completes in < 200ms
```

**Technical Notes:**
- Index type: IVFFlat (pgvector)
- Distance metric: Cosine distance
- Endpoint: GET /api/v1/similar/{artist}?limit=5

**Definition of Done:**
- [ ] DatabaseManager.find_similar_artists() uses vector index
- [ ] GET /api/v1/similar/{artist} endpoint implemented
- [ ] Returns artists ordered by similarity (cosine distance)
- [ ] Excludes the query artist from results
- [ ] Query performance optimized with IVFFlat index
- [ ] Unit tests for similarity logic pass
- [ ] Integration test with PostgreSQL + pgvector passes
- [ ] API documentation updated
- [ ] Code reviewed and merged

---

### Story E3.S5: Redis Caching Layer

**As a** system
**I want** to cache frequently accessed StyleProfiles in Redis
**So that** retrieval is sub-millisecond

**Acceptance Criteria:**

```gherkin
Feature: Redis Caching
  Background:
    Given Redis is running on localhost:6379
    And a StyleProfile for "J Dilla" exists in PostgreSQL

  Scenario: Cache miss - load from database
    Given the profile is not in Redis cache
    When I call cached_db.get_style_profile("J Dilla")
    Then it queries PostgreSQL
    And it stores result in Redis with key "profile:J Dilla"
    And it sets TTL to 7 days
    And it returns the StyleProfile

  Scenario: Cache hit - load from Redis
    Given the profile is cached in Redis
    When I call cached_db.get_style_profile("J Dilla")
    Then it retrieves from Redis (no PostgreSQL query)
    And it deserializes the JSON
    And it returns the StyleProfile
    And retrieval time < 10ms

  Scenario: Invalidate cache on profile update
    Given the profile is cached
    When I call db.save_style_profile(updated_profile)
    Then it updates PostgreSQL
    And it deletes Redis key "profile:J Dilla"
    And next get() loads fresh data from PostgreSQL

  Scenario: Cache eviction after TTL
    Given the profile was cached 8 days ago
    When the TTL expires
    Then Redis evicts the key
    And next get() loads from PostgreSQL
    And re-caches with new 7-day TTL
```

**Technical Notes:**
- Library: redis-py 5.0+
- TTL: 7 days (604800 seconds)
- Key pattern: `profile:{artist_name}`
- Serialization: JSON

**Definition of Done:**
- [ ] Redis client configured and connected
- [ ] DatabaseManager.get_style_profile() checks Redis first
- [ ] Cache miss loads from PostgreSQL and caches result
- [ ] Cache hit returns immediately from Redis
- [ ] save_style_profile() invalidates cache
- [ ] TTL set to 7 days
- [ ] Unit tests with mock Redis pass
- [ ] Integration test with real Redis passes
- [ ] Code reviewed and merged

---

### Story E3.S6: Generation History Analytics

**As a** product manager
**I want** to track generation metrics
**So that** I can optimize provider selection and costs

**Acceptance Criteria:**

```gherkin
Feature: Generation History
  Background:
    Given 100 generations have been completed:
      | Provider   | Count | Avg Time (ms) | Total Cost |
      | anthropic  | 60    | 1800          | $0.60      |
      | google     | 30    | 1200          | $0.15      |
      | openai     | 10    | 2100          | $0.12      |

  Scenario: Query generation statistics
    Given the GET /api/v1/stats endpoint
    When I GET /api/v1/stats
    Then the response contains:
      | Metric                  | Value     |
      | total_generations       | 100       |
      | avg_generation_time_ms  | 1670      |
      | total_cost_usd          | 0.87      |
      | provider_usage          | {"anthropic": 60, "google": 30, "openai": 10} |
      | avg_cost_per_generation | 0.0087    |

  Scenario: Query artist-specific history
    Given 10 generations for "J Dilla"
    When I GET /api/v1/artists/J%20Dilla/history
    Then the response contains list of 10 generations
    And each entry includes:
      | Field              | Example Value               |
      | provider_used      | anthropic                   |
      | generation_time_ms | 1847                        |
      | tokens_used        | 2340                        |
      | cost_usd           | 0.0123                      |
      | user_params        | {"bars": 4, "tempo": 95}    |
      | created_at         | 2025-11-15T10:30:00Z        |

  Scenario: Identify cost optimization opportunities
    Given the stats show Gemini costs 10x less than Claude
    And Gemini has 100% success rate
    When analyzing the data
    Then recommend switching primary provider to Gemini
    Or recommend using Gemini for simple requests, Claude for complex
```

**Technical Notes:**
- Endpoint: GET /api/v1/stats
- Endpoint: GET /api/v1/artists/{artist}/history
- Queries: Aggregate generation_history table

**Definition of Done:**
- [ ] GET /api/v1/stats endpoint implemented
- [ ] GET /api/v1/artists/{artist}/history endpoint implemented
- [ ] Aggregates generation_history table
- [ ] Returns total generations, avg time, total cost
- [ ] Returns provider usage breakdown
- [ ] Unit tests for aggregation queries pass
- [ ] Integration test with sample data passes
- [ ] API documentation updated
- [ ] Code reviewed and merged

---

## Epic 4: API Layer

**Goal:** Provide RESTful API endpoints for all system operations accessible from Max for Live and CLI.

**User Value:** Users interact with the system through a well-documented, reliable API with proper error handling and validation.

**Dependencies:** Epic 1 (Research), Epic 2 (Generation), Epic 3 (Database)

**Acceptance Criteria:**
- WHEN a client makes an API request
- THEN the request is validated using Pydantic models
- AND errors return clear, actionable messages
- AND successful responses follow consistent format
- AND all endpoints are documented in OpenAPI/Swagger

---

### Story E4.S1: FastAPI Application Setup

**As a** developer
**I want** to set up the FastAPI application with middleware
**So that** the API is production-ready

**Acceptance Criteria:**

```gherkin
Feature: FastAPI Setup
  Scenario: Initialize FastAPI app
    Given the src/api/main.py file
    Then it creates FastAPI app with title "MidiDrumiGen API"
    And it sets version "2.0.0"
    And it enables CORS for localhost (development)
    And it includes routers: research, generate, utils
    And it adds exception handlers
    And it enables OpenAPI docs at /docs

  Scenario: Configure CORS for development
    Given the CORS middleware
    Then it allows origins: ["http://localhost:*"]
    And it allows all methods
    And it allows all headers
    And it allows credentials

  Scenario: Add global exception handler
    Given an unhandled exception occurs
    When the API processes a request
    Then it catches the exception
    And it returns 500 Internal Server Error
    And it includes error details in response:
      | Field       | Value                              |
      | error       | internal_server_error              |
      | message     | An unexpected error occurred       |
      | request_id  | <UUID>                             |
      | timestamp   | 2025-11-18T10:30:00Z               |
    And it logs the full stack trace

  Scenario: Health check endpoint
    Given the GET /health endpoint
    When I GET /health
    Then the response status is 200 OK
    And the response body contains:
      | Field      | Value    |
      | status     | healthy  |
      | version    | 2.0.0    |
      | database   | connected|
      | redis      | connected|
      | celery     | active   |
```

**Technical Notes:**
- File: `src/api/main.py`
- Framework: FastAPI 0.109.2+
- Middleware: CORS, exception handling
- Documentation: Auto-generated OpenAPI/Swagger

**Definition of Done:**
- [ ] FastAPI app created in src/api/main.py
- [ ] CORS middleware configured
- [ ] Global exception handler added
- [ ] Health check endpoint implemented
- [ ] All routers included
- [ ] OpenAPI documentation enabled at /docs
- [ ] Server starts without errors
- [ ] Health check returns 200
- [ ] Code reviewed and merged

---

### Story E4.S2: Research Endpoints

**As a** API consumer
**I want** to trigger and monitor research via REST endpoints
**So that** I can research artists from any client

**Acceptance Criteria:**

```gherkin
Feature: Research Endpoints
  Scenario: Trigger research (already implemented in E1.S7)
    Given the POST /api/v1/research endpoint exists
    When I POST /api/v1/research with {"artist": "Elvin Jones"}
    Then the response status is 202 Accepted
    And it returns task_id for polling

  Scenario: Check if artist is cached (already implemented in E1.S7)
    Given the GET /api/v1/research/{artist} endpoint exists
    When I GET /api/v1/research/Travis%20Barker
    Then it returns cache status and confidence

  Scenario: Augment artist (already implemented in E1.S8)
    Given the POST /api/v1/augment/{artist} endpoint exists
    When I POST /api/v1/augment/J%20Dilla
    Then it triggers augmentation task
```

**Technical Notes:**
- File: `src/api/routes/research.py`
- Endpoints: Implemented in E1.S7 and E1.S8

**Definition of Done:**
- [ ] Already completed in Epic 1 stories
- [ ] Verify endpoints work end-to-end
- [ ] Integration test suite passes

---

### Story E4.S3: Generation Endpoints

**As a** API consumer
**I want** to generate MIDI patterns via REST endpoint
**So that** I can create patterns from any client

**Acceptance Criteria:**

```gherkin
Feature: Generation Endpoints
  Background:
    Given the POST /api/v1/generate endpoint
    And a StyleProfile for "John Bonham" exists

  Scenario: Generate patterns for cached artist
    Given the artist "John Bonham" is cached
    When I POST /api/v1/generate with:
      | Field           | Value            |
      | artist          | John Bonham      |
      | bars            | 4                |
      | tempo           | 120              |
      | time_signature  | [4, 4]           |
      | variations      | 4                |
      | provider        | auto             |
    Then the response status is 200 OK
    And the response body contains:
      | Field              | Value                              |
      | status             | success                            |
      | generation_time_ms | < 120000 (2 min)                   |
      | midi_files         | [path1, path2, path3, path4]       |
      | provider_used      | anthropic or google or openai      |
      | confidence         | 0.89                               |

  Scenario: Generate with specific LLM provider
    Given I want to use Claude specifically
    When I POST /api/v1/generate with provider="anthropic"
    Then the system uses Claude 4.5 Sonnet
    And it does not fallback to other providers (unless Claude fails)

  Scenario: Artist not researched error
    Given the artist "Unknown Artist" is not in database
    When I POST /api/v1/generate with artist="Unknown Artist"
    Then the response status is 404 Not Found
    And the response body contains:
      | Field          | Value                                      |
      | error          | artist_not_found                           |
      | message        | Artist not researched. Please research first.|
      | suggest_endpoint| /api/v1/research                          |

  Scenario: Invalid parameters error
    Given invalid parameters: bars=0, tempo=5000
    When I POST /api/v1/generate
    Then the response status is 422 Unprocessable Entity
    And the response body contains validation errors:
      | Field | Error                              |
      | bars  | Value must be between 1 and 16     |
      | tempo | Value must be between 40 and 300   |
```

**Technical Notes:**
- File: `src/api/routes/generate.py`
- Endpoint: POST /api/v1/generate
- Validation: Pydantic models
- Response: Synchronous (generation completes before response)

**Definition of Done:**
- [ ] POST /api/v1/generate endpoint implemented
- [ ] Validates request using Pydantic model
- [ ] Loads StyleProfile from database
- [ ] Calls HybridCoordinator.generate()
- [ ] Saves generation history
- [ ] Returns MIDI file paths and metadata
- [ ] Returns 404 if artist not cached
- [ ] Returns 422 for invalid parameters
- [ ] Unit tests for endpoint logic pass
- [ ] Integration test with full pipeline passes
- [ ] API documentation updated
- [ ] Code reviewed and merged

---

### Story E4.S4: Utility Endpoints

**As a** API consumer
**I want** utility endpoints for listing artists and checking stats
**So that** I can browse cached artists and monitor usage

**Acceptance Criteria:**

```gherkin
Feature: Utility Endpoints
  Scenario: List all cached artists
    Given the GET /api/v1/artists endpoint
    And 100 artists are cached in database
    When I GET /api/v1/artists
    Then the response status is 200 OK
    And the response body contains:
      | Field       | Value    |
      | total       | 100      |
      | cached      | 100      |
      | researching | 0        |
      | failed      | 0        |
    And it includes recent array with 10 most recently updated artists

  Scenario: Paginate artist list
    Given 500 artists in database
    When I GET /api/v1/artists?page=2&limit=50
    Then the response contains artists 51-100
    And it includes pagination metadata:
      | Field       | Value |
      | page        | 2     |
      | limit       | 50    |
      | total_pages | 10    |
      | total       | 500   |

  Scenario: Search artists by name
    Given artists including "John Bonham", "Bonnie Raitt", "Questlove"
    When I GET /api/v1/artists?search=bon
    Then the response contains:
      | Artist        |
      | John Bonham   |
      | Bonnie Raitt  |
    And it excludes Questlove

  Scenario: Get system statistics (already implemented in E3.S6)
    Given the GET /api/v1/stats endpoint exists
    When I GET /api/v1/stats
    Then it returns generation metrics

  Scenario: Task status polling (already referenced in E1.S7)
    Given the GET /api/v1/task/{task_id} endpoint
    When I poll task status
    Then it returns current status, progress, result
```

**Technical Notes:**
- File: `src/api/routes/utils.py`
- Endpoints: GET /api/v1/artists, GET /api/v1/stats (already in E3.S6), GET /api/v1/task/{task_id}
- Pagination: page, limit query parameters

**Definition of Done:**
- [ ] GET /api/v1/artists endpoint implemented
- [ ] Supports pagination (page, limit)
- [ ] Supports search (name filtering)
- [ ] Returns total, cached, researching, failed counts
- [ ] GET /api/v1/task/{task_id} implemented (if not already)
- [ ] Unit tests for endpoints pass
- [ ] Integration tests pass
- [ ] API documentation updated
- [ ] Code reviewed and merged

---

### Story E4.S5: Request Validation with Pydantic

**As a** developer
**I want** to validate all API requests using Pydantic models
**So that** invalid data is rejected with clear error messages

**Acceptance Criteria:**

```gherkin
Feature: Request Validation
  Scenario: Define GenerateRequest model
    Given the GenerateRequest Pydantic model
    Then it validates:
      | Field           | Type        | Constraints               |
      | artist          | str         | 1-100 chars, alphanumeric |
      | bars            | int         | 1-16                      |
      | tempo           | int         | 40-300                    |
      | time_signature  | Tuple[int, int] | (3,4), (4,4), (5,4), etc. |
      | variations      | int         | 1-8                       |
      | provider        | Optional[str] | "auto", "anthropic", "google", "openai" |
      | humanize        | bool        | Default True              |

  Scenario: Reject invalid artist name
    Given I POST /api/v1/generate with artist=""
    When FastAPI validates the request
    Then it returns 422 Unprocessable Entity
    And the error details include:
      | Field   | Error                              |
      | artist  | ensure this value has at least 1 character |

  Scenario: Reject out-of-range tempo
    Given I POST /api/v1/generate with tempo=500
    When FastAPI validates the request
    Then it returns 422 Unprocessable Entity
    And the error details include:
      | Field  | Error                            |
      | tempo  | Value must be between 40 and 300 |

  Scenario: Define ResearchRequest model
    Given the ResearchRequest Pydantic model
    Then it validates:
      | Field   | Type        | Constraints        |
      | artist  | str         | 1-100 chars        |
      | depth   | Optional[str] | "quick" or "full" |

  Scenario: Automatic OpenAPI schema generation
    Given Pydantic models are defined
    When I access /docs
    Then OpenAPI schema includes all models
    And it shows constraints and examples
    And it enables "Try it out" functionality
```

**Technical Notes:**
- Library: Pydantic 2.6+
- Models: GenerateRequest, ResearchRequest, etc.
- Location: `src/api/models.py`

**Definition of Done:**
- [ ] GenerateRequest Pydantic model defined
- [ ] ResearchRequest Pydantic model defined
- [ ] All fields have type hints and validators
- [ ] Custom validators for complex fields (e.g., time_signature)
- [ ] OpenAPI schema auto-generated from models
- [ ] /docs shows all models with examples
- [ ] Unit tests for validation logic pass
- [ ] Code reviewed and merged

---

### Story E4.S6: Error Handling and Responses

**As a** API consumer
**I want** consistent error response format
**So that** I can handle errors programmatically

**Acceptance Criteria:**

```gherkin
Feature: Error Responses
  Scenario: Standard error response format
    Given any error occurs in the API
    Then the response body follows format:
      | Field       | Type   | Description                     |
      | error       | string | Machine-readable error code     |
      | message     | string | Human-readable error message    |
      | details     | object | Optional additional information |
      | timestamp   | string | ISO 8601 timestamp              |
      | request_id  | string | UUID for tracing                |

  Scenario: Validation error (422)
    Given invalid request parameters
    When the API validates the request
    Then the response status is 422 Unprocessable Entity
    And the response body contains:
      | Field    | Value                              |
      | error    | validation_error                   |
      | message  | Request validation failed          |
      | details  | {field-level errors from Pydantic} |

  Scenario: Not found error (404)
    Given artist not in database
    When the API processes the request
    Then the response status is 404 Not Found
    And the response body contains:
      | Field    | Value                                 |
      | error    | artist_not_found                      |
      | message  | Artist not researched                 |
      | details  | {suggest_endpoint: /api/v1/research}  |

  Scenario: Internal server error (500)
    Given an unexpected exception occurs
    When the API processes the request
    Then the response status is 500 Internal Server Error
    And the response body contains:
      | Field    | Value                          |
      | error    | internal_server_error          |
      | message  | An unexpected error occurred   |
    And the full stack trace is logged (not exposed to client)

  Scenario: Rate limiting error (429)
    Given the user exceeds rate limit (100 req/hour)
    When the API processes the request
    Then the response status is 429 Too Many Requests
    And the response body contains:
      | Field    | Value                               |
      | error    | rate_limit_exceeded                 |
      | message  | Too many requests, please retry later |
      | details  | {retry_after: 3600 seconds}         |
```

**Technical Notes:**
- File: `src/api/error_handlers.py`
- Exception hierarchy: MidiDrumiGenError → ResearchError, GenerationError, etc.
- Logging: All errors logged with request_id for tracing

**Definition of Done:**
- [ ] Standard error response format defined
- [ ] Exception handler for validation errors (422)
- [ ] Exception handler for not found (404)
- [ ] Exception handler for internal errors (500)
- [ ] Exception handler for rate limiting (429)
- [ ] All errors include request_id for tracing
- [ ] Errors logged with full context
- [ ] Unit tests for error handlers pass
- [ ] Code reviewed and merged

---

### Story E4.S7: Rate Limiting

**As a** system administrator
**I want** to rate limit API requests
**So that** the system isn't overwhelmed by abusive clients

**Acceptance Criteria:**

```gherkin
Feature: Rate Limiting
  Background:
    Given rate limit is 100 requests per hour per IP

  Scenario: Allow requests within limit
    Given a client has made 50 requests in the last hour
    When the client makes another request
    Then the request is processed normally
    And the response header includes:
      | Header                | Value |
      | X-RateLimit-Limit     | 100   |
      | X-RateLimit-Remaining | 49    |
      | X-RateLimit-Reset     | <timestamp> |

  Scenario: Reject requests exceeding limit
    Given a client has made 100 requests in the last hour
    When the client makes another request
    Then the response status is 429 Too Many Requests
    And the response body contains rate limit error
    And the response header includes:
      | Header                | Value       |
      | X-RateLimit-Limit     | 100         |
      | X-RateLimit-Remaining | 0           |
      | X-RateLimit-Reset     | <timestamp> |
      | Retry-After           | 3600        |

  Scenario: Rate limit resets after time window
    Given a client was rate limited 1 hour ago
    When the time window resets
    Then the client's request count resets to 0
    And new requests are allowed
```

**Technical Notes:**
- Implementation: Redis-based sliding window
- Library: slowapi or custom middleware
- Rate: 100 req/hour per IP (configurable)
- Headers: X-RateLimit-* headers

**Definition of Done:**
- [ ] Rate limiting middleware implemented
- [ ] Uses Redis for distributed rate limiting
- [ ] Returns 429 when limit exceeded
- [ ] Includes X-RateLimit-* headers
- [ ] Configurable rate limit (env var)
- [ ] Unit tests with mocked Redis pass
- [ ] Integration test with real Redis passes
- [ ] Code reviewed and merged

---

### Story E4.S8: API Documentation

**As a** API consumer
**I want** comprehensive API documentation
**So that** I can integrate with the API easily

**Acceptance Criteria:**

```gherkin
Feature: API Documentation
  Scenario: Access interactive documentation
    Given the FastAPI server is running
    When I navigate to http://localhost:8000/docs
    Then I see Swagger UI
    And it lists all endpoints grouped by tags:
      | Tag         | Endpoints                           |
      | Research    | POST /research, GET /research/{artist}, POST /augment/{artist} |
      | Generation  | POST /generate                      |
      | Utilities   | GET /artists, GET /similar/{artist}, GET /stats, GET /task/{task_id} |
      | Health      | GET /health                         |
    And each endpoint shows request/response schemas
    And I can "Try it out" with example requests

  Scenario: Access alternative documentation
    Given the FastAPI server is running
    When I navigate to http://localhost:8000/redoc
    Then I see ReDoc documentation
    And it provides detailed endpoint descriptions
    And it shows request/response examples

  Scenario: Download OpenAPI schema
    When I GET /openapi.json
    Then I receive the OpenAPI 3.0 schema as JSON
    And it includes all endpoints, models, and examples
    And I can import it into Postman or other API tools

  Scenario: Example requests in documentation
    Given the GenerateRequest model has examples
    Then the docs show example:
      | Field           | Example Value    |
      | artist          | J Dilla          |
      | bars            | 4                |
      | tempo           | 95               |
      | time_signature  | [4, 4]           |
      | variations      | 4                |
      | provider        | auto             |
      | humanize        | true             |
```

**Technical Notes:**
- Auto-generated: FastAPI OpenAPI
- UIs: Swagger UI (/docs), ReDoc (/redoc)
- Schema: OpenAPI 3.0 (/openapi.json)
- Examples: Add to Pydantic models using Config

**Definition of Done:**
- [ ] /docs (Swagger UI) is accessible
- [ ] /redoc (ReDoc) is accessible
- [ ] /openapi.json returns schema
- [ ] All endpoints documented with descriptions
- [ ] Request/response models have examples
- [ ] Tags used to group endpoints logically
- [ ] "Try it out" functionality works
- [ ] Documentation reviewed for clarity
- [ ] Code reviewed and merged

---

## Epic 5: MIDI Export & Humanization

**Goal:** Export LLM-generated MIDI with natural-sounding humanization that mimics human drummer imperfections.

**User Value:** Users get MIDI files that sound human and natural, not robotic or quantized.

**Dependencies:** Epic 2 (Generation produces MIDI JSON)

**Acceptance Criteria:**
- WHEN the system exports MIDI from LLM JSON output
- THEN it applies micro-timing variations (±5-20ms)
- AND it applies velocity variations (±10-20%)
- AND it adds ghost notes based on artist style
- AND it validates MIDI structure before export
- AND the MIDI file is compatible with Ableton Live and other DAWs

---

### Story E5.S1: JSON to MIDI Conversion

**As a** system
**I want** to convert LLM JSON output to standard MIDI files
**So that** patterns can be imported into DAWs

**Acceptance Criteria:**

```gherkin
Feature: JSON to MIDI Conversion
  Background:
    Given LLM JSON output:
      ```json
      {
        "notes": [
          {"pitch": 36, "velocity": 90, "time": 0, "duration": 120},
          {"pitch": 42, "velocity": 80, "time": 240, "duration": 120},
          {"pitch": 38, "velocity": 95, "time": 480, "duration": 120}
        ],
        "tempo": 120,
        "time_signature": [4, 4],
        "total_bars": 4
      }
      ```

  Scenario: Convert JSON to MIDI file
    Given the JSON output above
    When I call export_midi_from_llm(json_data, "John Bonham", 1, profile)
    Then it creates a MidiFile using mido
    And it sets tempo to 120 BPM
    And it sets time signature to 4/4
    And it creates a track with 3 note_on/note_off message pairs
    And it saves to output/john_bonham_var1_YYYYMMDD_HHMMSS.mid
    And it returns the file path

  Scenario: Set MIDI tempo correctly
    Given tempo 120 BPM
    When converting to MIDI
    Then it calculates tempo in microseconds: 500000 (60s / 120 BPM * 1000000)
    And it adds set_tempo meta message
    And DAWs interpret the file at 120 BPM

  Scenario: Set time signature correctly
    Given time signature [4, 4]
    When converting to MIDI
    Then it adds time_signature meta message with numerator=4, denominator=4
    And DAWs interpret the file as 4/4 time

  Scenario: Map notes to MIDI messages
    Given note {"pitch": 36, "velocity": 90, "time": 0, "duration": 120}
    When converting to MIDI
    Then it creates note_on message at time 0, note=36, velocity=90, channel=9
    And it creates note_off message at time 120, note=36, velocity=0, channel=9
    And channel 9 = channel 10 in 1-indexed GM drums

  Scenario: Calculate delta times correctly
    Given notes at times [0, 240, 480]
    When converting to MIDI
    Then note 1 has delta_time = 0
    And note 2 has delta_time = 240
    And note 3 has delta_time = 240 (480 - 240)
    And mido handles cumulative time correctly
```

**Technical Notes:**
- File: `src/midi/export.py`
- Function: `export_midi_from_llm(json_data, artist_name, variation_num, style_profile, output_dir)`
- Library: mido 1.3.3
- Channel: 9 (10 in 1-indexed, GM drums)
- Ticks per beat: 480 (standard)

**Definition of Done:**
- [ ] export_midi_from_llm() function implemented
- [ ] Creates MidiFile with proper tempo and time signature
- [ ] Converts JSON notes to note_on/note_off messages
- [ ] Uses channel 9 for drums
- [ ] Calculates delta times correctly
- [ ] Saves to organized output directory
- [ ] Returns file path
- [ ] Unit tests for conversion logic pass
- [ ] Integration test produces valid MIDI file
- [ ] MIDI file opens in Ableton Live and other DAWs
- [ ] Code reviewed and merged

---

### Story E5.S2: Micro-Timing Humanization

**As a** user
**I want** MIDI notes to have slight timing imperfections
**So that** the pattern sounds human, not robotic

**Acceptance Criteria:**

```gherkin
Feature: Micro-Timing Humanization
  Background:
    Given a note at time 480 (quarter note on beat 2)
    And micro_timing_ms parameter from StyleProfile

  Scenario: Apply micro-timing variation
    Given micro_timing_ms = ±10ms
    And ticks_per_beat = 480
    When applying humanization
    Then it generates random offset in range [-10ms, +10ms]
    And it converts ms to ticks: offset_ticks = (offset_ms / 1000) * (tempo / 60) * ticks_per_beat
    And it adds offset_ticks to note time
    And the note plays slightly early or late

  Scenario: Larger variation for laid-back drummers
    Given a StyleProfile for "J Dilla" with micro_timing_ms = ±20ms (laid-back feel)
    When applying humanization
    Then notes vary by up to ±20ms
    And it creates the characteristic "behind the beat" feel

  Scenario: Smaller variation for precise drummers
    Given a StyleProfile for "Travis Barker" with micro_timing_ms = ±5ms (precise)
    When applying humanization
    Then notes vary by only ±5ms
    And the pattern sounds tight and precise

  Scenario: Disable humanization option
    Given humanize=False in generation parameters
    When exporting MIDI
    Then it skips micro-timing variation
    And all notes are exactly quantized

  Scenario: Apply different variation to different drum types
    Given kick drums should be tighter (±5ms)
    And hi-hats can be looser (±15ms)
    When applying humanization
    Then it uses different ranges per drum type
    And it creates more realistic human feel
```

**Technical Notes:**
- File: `src/midi/humanize.py`
- Function: `apply_micro_timing(notes, tempo, ticks_per_beat, micro_timing_ms)`
- Algorithm: Random uniform distribution within ±range
- Per-drum variation: Optional enhancement

**Definition of Done:**
- [ ] apply_micro_timing() function implemented
- [ ] Converts ms offset to ticks based on tempo
- [ ] Applies random offset to each note time
- [ ] Range configurable per StyleProfile
- [ ] Option to disable humanization
- [ ] (Optional) Different ranges per drum type
- [ ] Unit tests for timing calculation pass
- [ ] Integration test verifies humanized MIDI sounds natural
- [ ] Code reviewed and merged

---

### Story E5.S3: Velocity Humanization

**As a** user
**I want** MIDI note velocities to vary naturally
**So that** the pattern has dynamic feel

**Acceptance Criteria:**

```gherkin
Feature: Velocity Humanization
  Background:
    Given a note with velocity 90
    And velocity_variation_percent from StyleProfile

  Scenario: Apply velocity variation
    Given velocity_variation_percent = 15% (±15%)
    When applying humanization
    Then it generates random variation in range [-15%, +15%]
    And it calculates new_velocity = velocity * (1 + variation)
    And it clamps to valid range [1, 127]
    And the note has varied velocity

  Scenario: Higher variation for expressive drummers
    Given a StyleProfile with velocity_variation_percent = 25% (very expressive)
    When applying humanization
    Then velocities vary significantly (±25%)
    And it creates dynamic, expressive feel

  Scenario: Lower variation for consistent drummers
    Given a StyleProfile with velocity_variation_percent = 5% (very consistent)
    When applying humanization
    Then velocities vary minimally (±5%)
    And it maintains uniform dynamics

  Scenario: Velocity accents on important beats
    Given a note on beat 1 (downbeat)
    When applying humanization
    Then it adds +10% velocity bonus for accents
    And downbeats stand out dynamically

  Scenario: Clamp velocity to valid MIDI range
    Given velocity 120 with +15% variation = 138
    When clamping to MIDI range
    Then it sets velocity to 127 (max)

  Scenario: Ghost notes have lower velocities
    Given a ghost note (secondary hi-hat hit)
    And ghost_note_velocity_range from StyleProfile
    When applying humanization
    Then it sets velocity to range [20, 50] (quiet)
    And it creates subtle ghost note texture
```

**Technical Notes:**
- File: `src/midi/humanize.py`
- Function: `apply_velocity_variation(notes, variation_percent, accent_beats)`
- Range: [1, 127] (MIDI valid range)
- Ghost notes: Separate velocity range

**Definition of Done:**
- [ ] apply_velocity_variation() function implemented
- [ ] Applies random percentage variation
- [ ] Clamps to [1, 127] range
- [ ] Adds accent bonus for downbeats
- [ ] Handles ghost notes with lower velocities
- [ ] Variation configurable per StyleProfile
- [ ] Unit tests for variation logic pass
- [ ] Integration test verifies dynamic feel
- [ ] Code reviewed and merged

---

### Story E5.S4: Ghost Note Addition

**As a** user
**I want** ghost notes added to patterns based on artist style
**So that** the pattern has realistic texture

**Acceptance Criteria:**

```gherkin
Feature: Ghost Note Addition
  Background:
    Given a drum pattern with kick and snare
    And ghost_note_probability from StyleProfile

  Scenario: Add ghost notes to hi-hat
    Given ghost_note_probability = 0.3 (30% chance)
    And a hi-hat note at time 240
    When evaluating ghost note insertion
    Then it checks if random(0, 1) < 0.3
    And 30% of the time it adds ghost note at offset time (e.g., time + 60 ticks)
    And ghost note has velocity in range [20, 50]

  Scenario: Artist with frequent ghost notes (Questlove)
    Given a StyleProfile with ghost_note_probability = 0.5 (50%)
    When adding ghost notes
    Then approximately 50% of beats get ghost notes
    And the pattern has rich, detailed texture

  Scenario: Artist with no ghost notes (electronic drums)
    Given a StyleProfile with ghost_note_probability = 0.0 (0%)
    When adding ghost notes
    Then no ghost notes are added
    And the pattern is clean and direct

  Scenario: Ghost notes offset from main notes
    Given a main hi-hat note at time 240
    And a ghost note is added
    Then it places ghost note at time 240 + 60 ticks (16th note offset)
    And it creates subtle rhythmic detail

  Scenario: Ghost notes on snare rim shots
    Given a snare note at time 480
    And ghost_note_probability = 0.4
    When adding ghost notes
    Then it may add rim shot (MIDI note 37) before/after snare
    And it creates realistic snare technique
```

**Technical Notes:**
- File: `src/midi/humanize.py`
- Function: `add_ghost_notes(notes, ghost_note_prob, ticks_per_beat)`
- Typical offsets: +60 ticks (32nd note), +30 ticks (64th note)
- Ghost velocity: [20, 50]

**Definition of Done:**
- [ ] add_ghost_notes() function implemented
- [ ] Randomly adds ghost notes based on probability
- [ ] Offsets ghost notes by small intervals
- [ ] Sets ghost note velocities to [20, 50]
- [ ] Probability configurable per StyleProfile
- [ ] Supports snare ghost notes (rim shots, flams)
- [ ] Unit tests for ghost note logic pass
- [ ] Integration test verifies ghost notes add texture
- [ ] Code reviewed and merged

---

### Story E5.S5: MIDI Validation

**As a** system
**I want** to validate MIDI structure before export
**So that** invalid MIDI is never exported

**Acceptance Criteria:**

```gherkin
Feature: MIDI Validation
  Scenario: Validate note pitch range
    Given a note with pitch 36 (kick)
    When validating the note
    Then it checks 35 <= pitch <= 81 (drum range)
    And validation passes

  Scenario: Reject invalid pitch
    Given a note with pitch 150 (invalid)
    When validating the note
    Then it raises ValidationError("Invalid pitch: must be 35-81 for drums")

  Scenario: Validate velocity range
    Given a note with velocity 90
    When validating the note
    Then it checks 1 <= velocity <= 127
    And validation passes

  Scenario: Reject invalid velocity
    Given a note with velocity 200 (invalid)
    When validating the note
    Then it raises ValidationError("Invalid velocity: must be 1-127")

  Scenario: Validate time values are non-negative
    Given a note with time -100 (invalid)
    When validating the note
    Then it raises ValidationError("Invalid time: must be >= 0")

  Scenario: Validate duration is positive
    Given a note with duration 0 (invalid)
    When validating the note
    Then it raises ValidationError("Invalid duration: must be > 0")

  Scenario: Validate tempo range
    Given tempo 500 BPM (too fast)
    When validating the MIDI data
    Then it raises ValidationError("Invalid tempo: must be 40-300 BPM")

  Scenario: Validate JSON structure
    Given MIDI JSON missing "notes" key
    When validating the structure
    Then it raises ValidationError("Missing required key: notes")

  Scenario: Comprehensive validation before export
    Given LLM JSON output
    When export_midi_from_llm() is called
    Then it validates structure, tempo, time signature, all notes
    And it only exports if all validations pass
    And it logs validation errors if any fail
```

**Technical Notes:**
- File: `src/midi/validate.py`
- Function: `validate_midi_json(midi_data)` → Optional[ValidationError]
- Checks: structure, pitch [35-81], velocity [1-127], time [≥0], duration [>0], tempo [40-300]

**Definition of Done:**
- [ ] validate_midi_json() function implemented
- [ ] Validates all required fields present
- [ ] Validates pitch range [35-81]
- [ ] Validates velocity range [1-127]
- [ ] Validates time/duration are valid
- [ ] Validates tempo range [40-300]
- [ ] Returns None if valid, ValidationError if invalid
- [ ] export_midi_from_llm() calls validation before export
- [ ] Unit tests for all validation rules pass
- [ ] Code reviewed and merged

---

## Epic 6: Ableton Integration

**Goal:** Enable seamless interaction with MidiDrumiGen from within Ableton Live via Max for Live device.

**User Value:** Users generate patterns directly in their DAW without leaving their creative workflow.

**Dependencies:** Epic 4 (API endpoints must exist)

**Acceptance Criteria:**
- WHEN a user loads the Max for Live device in Ableton
- THEN they can input artist name, set parameters, and generate patterns
- AND generated MIDI clips appear in clip slots automatically
- AND the UI shows real-time progress during research/generation
- AND the device handles errors gracefully with user-friendly messages

---

### Story E6.S1: Max for Live Device Shell

**As a** Ableton Live user
**I want** to load the MidiDrumiGen device in Ableton
**So that** I can access the generation functionality

**Acceptance Criteria:**

```gherkin
Feature: Max for Live Device
  Scenario: Load device in Ableton Live
    Given Ableton Live 11+ is open
    When I drag MidiDrumiGen.amxd to a MIDI track
    Then the device loads successfully
    And it displays the MidiDrumiGen UI
    And it doesn't crash or freeze Ableton

  Scenario: Device UI layout
    Given the device is loaded
    Then I see:
      | Component           | Type          |
      | Artist input field  | Text entry    |
      | Bars dropdown       | Dropdown      |
      | Tempo number box    | Number entry  |
      | Time signature      | Dropdown      |
      | Variations dropdown | Dropdown      |
      | Generate button     | Button        |
      | Augment button      | Button        |
      | Progress bar        | Progress      |
      | Status text         | Text display  |
      | Confidence indicator| Visual rating |

  Scenario: Default parameter values
    Given the device is freshly loaded
    Then default values are:
      | Parameter      | Default |
      | Artist         | (empty) |
      | Bars           | 4       |
      | Tempo          | 120     |
      | Time signature | 4/4     |
      | Variations     | 4       |

  Scenario: Device persistence
    Given I set artist to "J Dilla" and tempo to 95
    When I save the Ableton project
    And I close and reopen the project
    Then the device remembers artist="J Dilla" and tempo=95
    And parameters persist across sessions
```

**Technical Notes:**
- File: `ableton/MidiDrumiGen.amxd`
- Framework: Max for Live (Max 8.5+)
- Language: Max patching + JavaScript (js object)
- UI: Max UI objects (textedit, live.menu, live.button, etc.)

**Definition of Done:**
- [ ] MidiDrumiGen.amxd device created
- [ ] UI laid out with all required components
- [ ] Default values set correctly
- [ ] Device loads without errors in Ableton Live 11+
- [ ] Parameters persist when project is saved/loaded
- [ ] Device tested on Windows and macOS
- [ ] Code reviewed and merged

---

### Story E6.S2: JavaScript HTTP Bridge

**As a** Max for Live device
**I want** to communicate with the FastAPI backend
**So that** I can trigger research and generation

**Acceptance Criteria:**

```gherkin
Feature: HTTP Bridge
  Background:
    Given the FastAPI server is running at http://localhost:8000
    And the JavaScript bridge is loaded in Max

  Scenario: Send HTTP POST request
    Given artist="J Dilla", bars=4, tempo=95
    When the user clicks "Generate"
    Then the JS bridge sends POST to http://localhost:8000/api/v1/generate
    And the request body contains:
      ```json
      {
        "artist": "J Dilla",
        "bars": 4,
        "tempo": 95,
        "time_signature": [4, 4],
        "variations": 4,
        "provider": "auto",
        "humanize": true
      }
      ```
    And the request has Content-Type: application/json

  Scenario: Handle successful response
    Given the API returns 200 OK with MIDI file paths
    When the JS bridge receives the response
    Then it parses the JSON response
    And it extracts midi_files array
    And it sends file paths to Max patch for clip import

  Scenario: Handle error response
    Given the API returns 404 Not Found (artist not researched)
    When the JS bridge receives the response
    Then it parses the error JSON
    And it extracts error message: "Artist not researched. Please research first."
    And it displays error in status text
    And it suggests user click "Research" button

  Scenario: Handle network errors
    Given the FastAPI server is not running
    When the JS bridge attempts HTTP request
    Then it catches connection error
    And it displays: "Error: Cannot connect to MidiDrumiGen server. Please start the API server."
    And it logs the error to Max console

  Scenario: Timeout handling
    Given a research request takes > 2 minutes
    When the JS bridge waits for response
    Then it shows progress: "Research in progress... (may take up to 20 min)"
    And it doesn't timeout prematurely
```

**Technical Notes:**
- File: `ableton/js/bridge.js`
- Library: Max js object (uses XMLHttpRequest or fetch-like API)
- Endpoint: http://localhost:8000 (configurable)
- Timeout: 120 seconds for generation, 1200 seconds for research

**Definition of Done:**
- [ ] bridge.js JavaScript file created
- [ ] Sends POST requests to /api/v1/generate
- [ ] Sends POST requests to /api/v1/research
- [ ] Sends GET requests to /api/v1/task/{task_id} for polling
- [ ] Parses JSON responses
- [ ] Handles success and error responses
- [ ] Handles network errors gracefully
- [ ] Unit tests for JS logic (if possible)
- [ ] Integration test with API passes
- [ ] Code reviewed and merged

---

### Story E6.S3: MIDI Clip Import via Live API

**As a** Ableton Live user
**I want** generated MIDI clips to appear in clip slots automatically
**So that** I can immediately play and edit them

**Acceptance Criteria:**

```gherkin
Feature: MIDI Clip Import
  Background:
    Given the MidiDrumiGen device is on MIDI track 1
    And 4 MIDI files have been generated
    And clip slots 1-4 are empty

  Scenario: Import MIDI clips to clip slots
    Given MIDI file paths from API response
    When the device imports clips
    Then it creates clips in slots 1-4 of track 1
    And each clip contains the MIDI data from the file
    And clips are named:
      | Slot | Name                |
      | 1    | J Dilla - Variation 1 |
      | 2    | J Dilla - Variation 2 |
      | 3    | J Dilla - Variation 3 |
      | 4    | J Dilla - Variation 4 |
    And clips are immediately playable

  Scenario: Use Live API for clip creation
    Given Max for Live Live API access
    When creating clips
    Then it calls live.path to get track reference
    And it calls live.object to create clip_slot objects
    And it calls create_clip() method
    And it sets MIDI notes using set_notes() or similar

  Scenario: Handle existing clips in slots
    Given clip slot 1 already has a clip
    When importing new clips
    Then it either:
      - Skips slot 1 and uses slots 2-5, OR
      - Asks user to confirm overwrite, OR
      - Finds next available empty slots
    And it doesn't destroy user's existing clips without warning

  Scenario: Multi-track import for multiple variations
    Given 8 variations requested
    When importing clips
    Then it fills all clip slots on track 1 (8 slots)
    Or it wraps to track 2 if track 1 is full
    And all clips are accessible

  Scenario: Clip properties match generation parameters
    Given tempo=95, time_signature=4/4, bars=4
    When importing clips
    Then each clip has length of 4 bars
    And clip tempo matches project tempo (or is set to 95)
    And clips are time-aligned to Ableton's grid
```

**Technical Notes:**
- API: Live API 12.0+ (via live.object, live.path)
- Methods: create_clip(), set_notes(), clip properties
- MIDI parsing: Use mido to read .mid files, convert to Live API format

**Definition of Done:**
- [ ] Clip import function implemented in Max patch
- [ ] Uses Live API to create clips in clip slots
- [ ] Reads MIDI files using mido (via Python external or js)
- [ ] Sets clip names to "{Artist} - Variation {N}"
- [ ] Handles existing clips gracefully
- [ ] Clips are immediately playable after import
- [ ] Tested in Ableton Live 11+
- [ ] Code reviewed and merged

---

### Story E6.S4: Progress Reporting

**As a** user
**I want** to see real-time progress during research and generation
**So that** I know the system is working

**Acceptance Criteria:**

```gherkin
Feature: Progress Reporting
  Background:
    Given the MidiDrumiGen device is loaded

  Scenario: Show progress during research
    Given the user clicks "Generate" for uncached artist
    When the API triggers research
    Then the progress bar shows 0%
    And status text shows "Checking cache..."
    And the device polls GET /api/v1/task/{task_id} every 2 seconds
    And it updates progress bar and status:
      | Time | Progress | Status                    |
      | 0s   | 0%       | Starting research...      |
      | 10s  | 20%      | Collecting papers...      |
      | 30s  | 40%      | Scraping articles...      |
      | 50s  | 60%      | Analyzing audio...        |
      | 70s  | 80%      | Building profile...       |
      | 80s  | 100%     | Research complete!        |

  Scenario: Show progress during generation
    Given the user clicks "Generate" for cached artist
    When the API generates patterns
    Then the progress bar animates
    And status text shows "Generating patterns..."
    And generation completes quickly (< 2 min)
    And status text updates to "✓ Complete! Importing clips..."

  Scenario: Disable UI during operation
    Given research or generation is in progress
    Then the "Generate" button is disabled (grayed out)
    And parameter inputs are disabled
    And the user cannot start another operation
    And the UI prevents conflicting requests

  Scenario: Cancel long-running operation
    Given research has been running for 10 minutes
    When the user clicks "Cancel" button
    Then the device stops polling the API
    And it sends DELETE /api/v1/task/{task_id} (if supported)
    And it resets progress bar and status
    And it re-enables UI controls
```

**Technical Notes:**
- Polling interval: 2 seconds
- Endpoints: GET /api/v1/task/{task_id}
- Max objects: live.text for status, live.progress for progress bar
- Task states: pending, in_progress, completed, failed

**Definition of Done:**
- [ ] Progress bar updates during research/generation
- [ ] Status text shows current operation
- [ ] Device polls task status endpoint every 2 seconds
- [ ] UI disabled during operations
- [ ] Cancel button stops polling (optional task cancellation)
- [ ] Tested with both quick (cached) and slow (research) operations
- [ ] Code reviewed and merged

---

### Story E6.S5: Error Handling and User Feedback

**As a** user
**I want** clear error messages when something goes wrong
**So that** I know how to fix the problem

**Acceptance Criteria:**

```gherkin
Feature: Error Handling
  Scenario: Artist not researched error
    Given the user enters "Unknown Artist"
    And the artist is not in the database
    When generation is attempted
    Then the API returns 404 Not Found
    And the device displays:
      "Artist not researched. Click 'Research' to analyze this artist first (takes 5-20 min)."
    And it highlights the "Research" button

  Scenario: API server not running error
    Given the FastAPI server is not running
    When the device attempts HTTP request
    Then it catches connection error
    And it displays:
      "Cannot connect to MidiDrumiGen server. Please start the server:
      uvicorn src.api.main:app --reload"
    And it shows instructions in status text

  Scenario: Research failed error
    Given research fails due to insufficient data
    When the API returns error
    Then the device displays:
      "Research failed: Only 3 sources found (minimum 8 required).
      Artist may not have enough available data."
    And it suggests trying a more well-documented artist

  Scenario: LLM generation failed error
    Given all LLM providers fail
    When the API returns 500 Internal Server Error
    Then the device displays:
      "Generation failed: All LLM providers unavailable.
      Please check API keys or try again later."
    And it logs the error to Max console for debugging

  Scenario: Invalid parameters error
    Given the user enters tempo=5000
    When generation is attempted
    Then the API returns 422 Unprocessable Entity
    And the device displays:
      "Invalid tempo: must be 40-300 BPM"
    And it highlights the tempo field in red

  Scenario: Success feedback
    Given generation completes successfully
    When clips are imported
    Then the device displays:
      "✓ Success! 4 clips imported to track 1."
    And it shows confidence score: "Style confidence: ●●●●◐ (0.82)"
    And status text is green
```

**Technical Notes:**
- Error display: live.text object (red color for errors, green for success)
- Confidence indicator: Visual rating (●●●●●)
- Max console: post() for detailed error logging

**Definition of Done:**
- [ ] All API error responses handled (404, 422, 500, etc.)
- [ ] Network errors handled (connection refused, timeout)
- [ ] User-friendly error messages displayed
- [ ] Errors suggest corrective actions
- [ ] Success messages confirm completion
- [ ] Confidence score displayed after successful generation
- [ ] Errors logged to Max console for debugging
- [ ] Code reviewed and merged

---

### Story E6.S6: Configuration and Settings

**As a** user
**I want** to configure API endpoint and other settings
**So that** the device works with my setup

**Acceptance Criteria:**

```gherkin
Feature: Configuration
  Scenario: Configure API endpoint
    Given the device has a settings panel
    When I click "Settings" button
    Then I see a dialog with:
      | Setting     | Default             |
      | API URL     | http://localhost:8000 |
      | Timeout (s) | 120                 |
    And I can change the URL to a custom server
    And settings are saved in device state

  Scenario: Auto-detect API server
    Given the device loads for the first time
    When it initializes
    Then it pings http://localhost:8000/health
    And if successful, displays "✓ Server connected"
    And if failed, displays "✗ Server not running - please start server"

  Scenario: Refresh cached artist list
    Given the device has been open for a while
    When I click "Refresh Artists" button
    Then it fetches GET /api/v1/artists
    And it populates autocomplete suggestions with cached artist names
    And I can select from dropdown instead of typing

  Scenario: Export configuration
    Given I have configured custom settings
    When I save the Ableton project
    Then the settings are saved in the .amxd state
    And they persist across sessions

  Scenario: Reset to defaults
    Given I have custom settings
    When I click "Reset to Defaults"
    Then API URL resets to http://localhost:8000
    And all parameters reset to defaults
```

**Technical Notes:**
- Settings storage: Max pattr or pattrstorage
- Health check: GET /health on device load
- Autocomplete: GET /api/v1/artists, populate live.menu

**Definition of Done:**
- [ ] Settings panel implemented
- [ ] API URL configurable
- [ ] Timeout configurable
- [ ] Auto-detect server on load (health check)
- [ ] Autocomplete artist names from cached list (optional)
- [ ] Settings persist across sessions
- [ ] Reset to defaults button works
- [ ] Code reviewed and merged

---

## Implementation Priority

Based on user value and dependencies, the recommended implementation order is:

1. **Epic 3: Database & Caching** (Stories E3.S1-E3.S6)
   - Reason: Foundational, required by all other epics
   - Status: Partially complete (Phase 2), finish remaining stories

2. **Epic 1: Research Pipeline** (Stories E1.S1-E1.S8)
   - Reason: Enables StyleProfile creation
   - Dependency: Epic 3

3. **Epic 2: LLM Generation Engine** (Stories E2.S1-E2.S7)
   - Reason: Core value delivery
   - Dependency: Epic 1 (needs StyleProfiles)

4. **Epic 5: MIDI Export & Humanization** (Stories E5.S1-E5.S5)
   - Reason: Completes generation pipeline
   - Dependency: Epic 2 (needs LLM output)

5. **Epic 4: API Layer** (Stories E4.S1-E4.S8)
   - Reason: Exposes all functionality via REST
   - Dependency: Epics 1, 2, 5

6. **Epic 6: Ableton Integration** (Stories E6.S1-E6.S6)
   - Reason: End-user interface
   - Dependency: Epic 4 (needs API endpoints)

---

## Next Steps

1. Review and approve this epic breakdown
2. Prioritize Phase 3 implementation based on the order above
3. Create first sprint focusing on Epic 3 (remaining database stories)
4. Begin Epic 1 implementation in parallel (research pipeline)
5. Set up continuous integration for automated testing
6. Establish code review process for each story

---

**Document Status:** ✅ Ready for Review
**Total Stories:** 40
**Estimated Complexity:** Medium-High (LLM integration, Max for Live, multi-source research)
**Recommended Sprint Length:** 2 weeks
**Team Size:** 1-2 developers

---

Generated: 2025-11-18
BMAD Method: Epic/Story Decomposition
Framework: Behavior-Driven Development (BDD)
