# Product Requirements Document (PRD)
# MidiDrumiGen v2.0 - On-Demand Artist-Style MIDI Generation System

**Version:** 2.0.0
**Date:** 2025-11-17
**Status:** Planning Phase
**Authors:** Product Team
**Stakeholders:** Musicians, Producers, Ableton Live Users

---

## Executive Summary

MidiDrumiGen v2.0 is an intelligent MIDI drum pattern generation plugin for Ableton Live that allows users to input ANY artist or band name and receive authentic drum patterns in that artist's style within 2 minutes (for cached artists). The system leverages multi-source research (academic papers, articles, audio analysis, MIDI databases) combined with state-of-the-art LLM technology to generate musically accurate and stylistically coherent drum patterns.

**Key Innovation:** On-demand style research eliminates the need for pre-trained models, enabling unlimited artist support.

---

## Problem Statement

### Current Challenges
1. **Limited Artist Coverage:** Existing drum generation tools support only pre-trained styles (typically 3-10 artists)
2. **No Real-Time Research:** Users cannot generate patterns for arbitrary artists
3. **Inflexible Systems:** Adding new artists requires retraining models (hours/days)
4. **Poor Style Accuracy:** Generic patterns don't capture artist-specific nuances
5. **Workflow Friction:** Moving between research, generation, and DAW integration is manual

### Target Users
- **Primary:** Music producers using Ableton Live 11+ for hip-hop, electronic, pop production
- **Secondary:** Drummers seeking inspiration in specific artist styles
- **Tertiary:** Music educators teaching drumming styles

---

## Product Vision

**Vision Statement:**
"Enable any musician to generate authentic drum patterns in any artist's style, instantly accessible within their DAW, powered by comprehensive multi-source research and AI generation."

**Success Metrics:**
- 10,000+ unique artists researched within first 6 months
- 95%+ user satisfaction with style accuracy (survey-based)
- < 2 min average generation time for cached artists
- 1,000+ monthly active users within first year
- 50%+ user retention after 30 days

---

## Product Objectives

### Phase 1 Objectives (MVP - 3 months)
1. ✓ Functional research pipeline for 100+ artists
2. ✓ Multi-provider LLM generation (OpenAI, Claude, Gemini)
3. ✓ PostgreSQL database with caching
4. ✓ Basic Max for Live device
5. ✓ MIDI export to Ableton clip slots

### Phase 2 Objectives (3-6 months)
1. Expand to 1,000+ cached artists
2. Implement augmentation feature (add more sources)
3. Add similarity search ("artists like [X]")
4. Improve generation quality (A/B testing)
5. Add VST3 version for other DAWs

### Phase 3 Objectives (6-12 months)
1. Local LLM support (privacy-focused users)
2. Real-time MIDI generation during playback
3. Style blending (combine multiple artists)
4. Community-contributed research sources
5. Mobile companion app (iOS/Android)

---

## User Stories

### Epic 1: Artist Research
**US-001:** As a producer, I want to input any artist name and have the system research their drumming style automatically, so I don't have to manually analyze tracks.

**Acceptance Criteria:**
- Input field accepts any text string
- System searches multiple source types (papers, articles, audio, MIDI)
- Progress bar shows research status
- Research completes within 20 minutes for first-time artists
- Results are cached for future use

**Priority:** P0 (Critical)

---

**US-002:** As a power user, I want to augment existing artist research with additional sources, so I can improve generation quality over time.

**Acceptance Criteria:**
- "Augment" button available for cached artists
- System collects 5+ additional sources per run
- Updated StyleProfile reflects new data
- Confidence score increases
- Generation quality improves measurably

**Priority:** P1 (High)

---

### Epic 2: Pattern Generation
**US-003:** As a producer, I want to generate 4-8 drum pattern variations in under 2 minutes, so I can quickly audition options during my creative session.

**Acceptance Criteria:**
- Generation completes < 2 min for cached artists
- Outputs 4-8 distinct variations
- Each variation is 1-16 bars (user-selectable)
- Tempo is configurable (40-300 BPM)
- Patterns are musically coherent

**Priority:** P0 (Critical)

---

**US-004:** As a musician, I want generated patterns to authentically reflect the input artist's style, so they fit my creative vision.

**Acceptance Criteria:**
- Tempo matches artist's typical range (±10 BPM tolerance)
- Swing/syncopation matches documented style
- Ghost notes present when characteristic of artist
- Velocity dynamics match artist's playing
- User survey: 80%+ rate as "accurate" or "very accurate"

**Priority:** P0 (Critical)

---

**US-005:** As a user, I want to specify generation parameters (bars, tempo, variations), so I have control over the output.

**Acceptance Criteria:**
- Bars: 1-16 (default: 4)
- Tempo: 40-300 BPM (default: artist's typical)
- Variations: 1-8 (default: 4)
- Time signature: 4/4, 3/4, 5/4, 6/8, 7/8
- Parameters persist between sessions

**Priority:** P1 (High)

---

### Epic 3: Ableton Integration
**US-006:** As an Ableton Live user, I want generated MIDI clips to automatically appear in my session, so I can immediately start working with them.

**Acceptance Criteria:**
- Max for Live device loads without errors
- Generated clips appear in clip slots on selected track
- Clips are named: "[Artist] - Variation [N]"
- MIDI is immediately editable
- Clips can be dragged to other tracks

**Priority:** P0 (Critical)

---

**US-007:** As a user, I want real-time feedback on research and generation progress, so I know the system is working.

**Acceptance Criteria:**
- Progress bar shows 0-100% completion
- Status text updates at each stage:
  - "Checking cache..."
  - "Researching [artist]... (5-20 min)"
  - "Generating patterns..."
  - "✓ Complete!"
- Error messages are clear and actionable
- Can cancel long-running operations

**Priority:** P1 (High)

---

### Epic 4: Quality & Performance
**US-008:** As a user, I want the system to gracefully handle errors, so my workflow isn't disrupted.

**Acceptance Criteria:**
- LLM provider failures trigger automatic fallback
- "Artist not found" returns helpful suggestions
- Rate limits are handled with retry logic
- Network errors show clear messages
- System never crashes Ableton

**Priority:** P0 (Critical)

---

**US-009:** As a user, I want generated MIDI to sound human and natural, not robotic.

**Acceptance Criteria:**
- Timing micro-variations (±5-20ms)
- Velocity variations (±10-20%)
- Ghost notes at appropriate probabilities
- Swing applied per artist style
- User survey: 75%+ rate as "natural" or "very natural"

**Priority:** P1 (High)

---

## Functional Requirements

### FR-1: Research Pipeline

#### FR-1.1: Scholar Paper Collection
- **Description:** Search academic databases for papers analyzing artist's drumming style
- **Sources:** Semantic Scholar API, arXiv, CrossRef
- **Output:** Text descriptions, tempo data, stylistic features
- **Minimum:** 3 papers per artist (if available)
- **Timeout:** 5 minutes

#### FR-1.2: Web Article Collection
- **Description:** Scrape music journalism for interviews, reviews, style analyses
- **Sources:** Pitchfork, Rolling Stone, Drummerworld, Wikipedia
- **Output:** Text excerpts, equipment mentions, style descriptors
- **Minimum:** 5 articles per artist
- **Timeout:** 5 minutes

#### FR-1.3: Audio Analysis
- **Description:** Analyze audio recordings for rhythm features
- **Sources:** YouTube (via yt-dlp), SoundCloud
- **Analysis:** Tempo, swing, syncopation, velocity distribution
- **Output:** Quantitative parameters (JSON)
- **Minimum:** 3 audio tracks per artist
- **Timeout:** 8 minutes

#### FR-1.4: MIDI Database Search
- **Description:** Find existing MIDI files of artist's songs
- **Sources:** BitMIDI, FreeMIDI, Musescore
- **Output:** MIDI pattern templates
- **Minimum:** 2 MIDI files per artist (if available)
- **Timeout:** 2 minutes

#### FR-1.5: Style Profile Generation
- **Description:** Aggregate all research into unified StyleProfile
- **Components:**
  - Text description (for LLM prompts)
  - Quantitative parameters (tempo, swing, velocities)
  - MIDI templates (reference patterns)
  - Vector embedding (for similarity search)
  - Confidence score (0.0-1.0)
- **Validation:** Confidence score must be ≥ 0.6 to be usable

---

### FR-2: Generation Engine

#### FR-2.1: Multi-Provider LLM Support
- **Providers:** OpenAI (GPT-4), Anthropic (Claude 3), Google (Gemini 1.5)
- **Fallback Logic:** If primary fails, try fallback providers sequentially
- **API Keys:** Stored in environment variables
- **Rate Limiting:** Implement exponential backoff
- **Cost Tracking:** Log API costs per generation

#### FR-2.2: Prompt Engineering
- **System Prompt:** Defines MIDI JSON format, drum mappings, rules
- **User Prompt:** Includes artist name, style characteristics, parameters
- **Few-Shot Examples:** Include 1-2 MIDI templates from research
- **Output Format:** Structured JSON with notes array
- **Validation:** Ensure output matches expected schema

#### FR-2.3: Template-Based Generation (Fallback)
- **Trigger:** When LLM generation fails or produces invalid output
- **Method:** Load MIDI templates, apply variations based on parameters
- **Variations:** Adjust swing, add ghost notes, vary velocities
- **Output:** Valid MIDI in artist's style

#### FR-2.4: Hybrid Coordination
- **Strategy:** Try LLM first, fall back to templates if needed
- **Validation:** Check MIDI validity (note ranges, velocities, timing)
- **Post-Processing:** Apply humanization to all outputs
- **Variations:** Generate 4-8 distinct patterns

---

### FR-3: Database & Caching

#### FR-3.1: Artist Registry
- **Schema:** id, name, research_status, last_updated, sources_count, confidence_score
- **Indexing:** Indexed on name for fast lookup
- **Uniqueness:** Artist names are unique (case-insensitive)

#### FR-3.2: Research Sources Storage
- **Schema:** id, artist_id, source_type, url, raw_content, extracted_data, confidence
- **Relationships:** Many-to-one with Artist
- **Retention:** Keep all sources indefinitely (for augmentation)

#### FR-3.3: Style Profiles
- **Schema:** id, artist_id, text_description, quantitative_params, midi_templates_json, embedding (vector), confidence_score
- **Uniqueness:** One profile per artist (updated on augmentation)
- **Vector Search:** Support similarity search via pgvector

#### FR-3.4: Generation History
- **Schema:** id, artist_id, provider_used, generation_time_ms, user_params, output_files
- **Purpose:** Track usage, performance, provider success rates
- **Retention:** Keep last 1000 generations per artist

---

### FR-4: Ableton Integration

#### FR-4.1: Max for Live Device
- **Requirements:** Max 8.5+, Ableton Live 11+
- **UI Components:**
  - Text input (artist name)
  - Generate button
  - Progress bar
  - Parameter controls (bars, tempo, variations)
  - Status display
  - Augment button
- **API Communication:** HTTP requests to localhost FastAPI server
- **Clip Import:** Use Live API to create clips in clip slots

#### FR-4.2: MIDI Clip Creation
- **Format:** Standard MIDI Type 0 files
- **Channel:** Channel 10 (GM drums)
- **Naming:** "[Artist] - Variation [N]"
- **Placement:** Fill next available clip slots on selected track
- **Undo Support:** Clips can be removed via Ableton's undo

---

### FR-5: API Endpoints

#### FR-5.1: Research Endpoints
```
POST /api/v1/research
  Request: {"artist": "John Bonham"}
  Response: {"task_id": "uuid", "status": "researching"}

GET /api/v1/research/{artist}
  Response: {"exists": true, "confidence": 0.85, "last_updated": "..."}

POST /api/v1/augment/{artist}
  Request: {}
  Response: {"task_id": "uuid", "status": "augmenting"}

GET /api/v1/task/{task_id}
  Response: {"status": "completed", "progress": 100, "result": {...}}
```

#### FR-5.2: Generation Endpoints
```
POST /api/v1/generate
  Request: {
    "artist": "John Bonham",
    "bars": 4,
    "tempo": 120,
    "time_signature": [4, 4],
    "variations": 4,
    "provider": "auto"  # or specific provider
  }
  Response: {
    "status": "success",
    "generation_time_ms": 1847,
    "midi_files": ["path/to/var1.mid", ...],
    "provider_used": "openai"
  }
```

#### FR-5.3: Utility Endpoints
```
GET /api/v1/artists
  Response: {"total": 1523, "cached": 1523, "researching": 0}

GET /api/v1/similar/{artist}
  Response: {"similar_artists": ["Artist A", "Artist B", ...]}

GET /api/v1/stats
  Response: {"total_generations": 45231, "avg_time_ms": 1920, ...}
```

---

## Non-Functional Requirements

### NFR-1: Performance
- **Research Time:** < 20 minutes for first-time artists (80th percentile)
- **Generation Time:** < 2 minutes for cached artists (95th percentile)
- **Database Queries:** < 100ms (average)
- **LLM API Calls:** < 30 seconds (average)
- **Concurrent Users:** Support 100 simultaneous users
- **Memory Usage:** < 2GB RAM per worker process

### NFR-2: Reliability
- **Uptime:** 99.5% availability (excluding planned maintenance)
- **Error Rate:** < 1% of generation requests fail
- **Data Durability:** PostgreSQL with daily backups
- **Fallback Success:** 95%+ of LLM failures resolved via fallback providers
- **Crash Recovery:** System auto-recovers from worker crashes

### NFR-3: Scalability
- **Artist Database:** Support 100,000+ cached artists
- **Horizontal Scaling:** Add Celery workers to handle load
- **Database Sharding:** Plan for future sharding by artist name hash
- **CDN for MIDI Files:** Use CDN for large-scale distribution (future)

### NFR-4: Security
- **API Keys:** Stored in environment variables, never in code
- **Input Validation:** Sanitize all user inputs
- **Rate Limiting:** 100 requests per user per hour
- **HTTPS:** All API communication over HTTPS in production
- **Database Access:** Read-only user for query endpoints

### NFR-5: Usability
- **Learning Curve:** New users can generate first pattern in < 5 minutes
- **Error Messages:** Clear, actionable, non-technical language
- **Accessibility:** UI follows WCAG 2.1 AA guidelines
- **Documentation:** Comprehensive user guide and API docs

### NFR-6: Maintainability
- **Code Coverage:** > 80% unit test coverage
- **Documentation:** All functions have docstrings
- **Logging:** Structured logging with log levels
- **Monitoring:** Prometheus metrics for key operations
- **Version Control:** Semantic versioning (SemVer)

---

## Technical Constraints

### TC-1: Dependencies
- **Python:** 3.11+ (latest stable)
- **PostgreSQL:** 15+ with pgvector extension
- **Redis:** 7.0+ for Celery task queue
- **Max for Live:** Requires Max 8.5+
- **Ableton Live:** 11+ (Suite or Standard with Max for Live)

### TC-2: External APIs
- **OpenAI:** Requires paid account, subject to rate limits
- **Anthropic:** Requires API access (may have waitlist)
- **Google:** Requires Gemini API access
- **Semantic Scholar:** Free with rate limits (100 req/5 min)
- **YouTube:** yt-dlp subject to YouTube's terms of service

### TC-3: Platform Support
- **Primary:** Windows 10/11, macOS 11+
- **Future:** Linux (for headless API server)
- **Ableton:** Max for Live device is macOS/Windows only

### TC-4: Storage
- **MIDI Files:** ~5KB per file, 8 variations = 40KB per generation
- **Audio Cache:** Not stored (analyzed and discarded)
- **Database:** ~500KB per artist (all sources + profile)
- **Estimate:** 10,000 artists = ~5GB database + ~400GB MIDI files (1M generations)

---

## User Interface Requirements

### UI-1: Max for Live Device

#### Layout
```
┌──────────────────────────────────────────┐
│  MidiDrumiGen v2.0                       │
├──────────────────────────────────────────┤
│  Artist: [_______________________]  [🔍] │
│                                          │
│  Status: Ready                           │
│  [████████████░░░░░░░░░░░░░] 60%        │
│                                          │
│  Bars: [ 4 ▼]  Tempo: [120]  Time: [4/4]│
│  Variations: [ 4 ▼]                      │
│                                          │
│  [  Generate  ]  [ Augment Research ]    │
│                                          │
│  Confidence: ●●●●◐ (0.82)               │
│  Cached: Yes | Last Updated: 2d ago      │
└──────────────────────────────────────────┘
```

#### UI Elements
1. **Artist Input:** Text field with autocomplete from cached artists
2. **Search Button:** Triggers research if not cached
3. **Status Display:** Shows current operation and progress
4. **Progress Bar:** Visual feedback for long operations
5. **Parameter Controls:** Dropdowns and number boxes
6. **Generate Button:** Primary action (disabled during generation)
7. **Augment Button:** Secondary action (only enabled if cached)
8. **Confidence Indicator:** Visual rating of StyleProfile quality
9. **Cache Status:** Shows if artist is cached and age

#### Color Scheme
- **Background:** Dark gray (#2B2B2B)
- **Text:** Light gray (#CCCCCC)
- **Accent:** Blue (#4A90E2)
- **Success:** Green (#7ED321)
- **Error:** Red (#D0021B)
- **Progress:** Gradient blue

---

### UI-2: Web Dashboard (Future Phase)

Optional web interface for power users:
- Browse cached artists
- View research sources
- Compare style profiles
- Track usage statistics
- Manage API keys
- Export data

---

## Success Metrics & KPIs

### User Engagement
- **DAU (Daily Active Users):** Target: 200 within 3 months
- **MAU (Monthly Active Users):** Target: 1,000 within 6 months
- **Session Length:** Average 15+ minutes per session
- **Retention:** 50%+ users return within 30 days

### System Performance
- **Research Success Rate:** > 90% of artists produce usable profiles
- **Generation Success Rate:** > 98% of requests complete successfully
- **Average Research Time:** < 12 minutes (first-time)
- **Average Generation Time:** < 90 seconds (cached)
- **Cache Hit Rate:** > 80% after first 3 months

### Quality Metrics
- **Style Accuracy (User Survey):** > 80% rate as "accurate" or better
- **Naturalness (User Survey):** > 75% rate as "natural" or better
- **Usefulness (User Survey):** > 85% would recommend to a friend
- **Confidence Score (System):** Average > 0.75 for cached artists

### Business Metrics (Future Commercial Release)
- **Free Trial Conversion:** > 15% convert to paid
- **Churn Rate:** < 5% monthly
- **NPS (Net Promoter Score):** > 50
- **Support Tickets:** < 0.5 per user per month

---

## Risk Assessment

### High-Risk Items
1. **LLM API Reliability:** Providers may have outages or rate limits
   - **Mitigation:** Multi-provider fallback, template-based backup
2. **Style Accuracy:** Generated patterns may not match user expectations
   - **Mitigation:** A/B testing, user feedback loop, augmentation feature
3. **Research Quality:** Some artists have limited available data
   - **Mitigation:** Confidence scoring, manual curation option
4. **Legal Issues:** Copyright concerns with MIDI database scraping
   - **Mitigation:** Use only public domain/CC-licensed sources

### Medium-Risk Items
1. **Ableton API Changes:** Live updates may break Max for Live device
   - **Mitigation:** Version pinning, automated testing
2. **Database Scaling:** 100K+ artists may stress PostgreSQL
   - **Mitigation:** Indexing, read replicas, future sharding
3. **User Adoption:** Musicians may prefer manual workflow
   - **Mitigation:** User research, onboarding improvements

### Low-Risk Items
1. **MIDI Export Compatibility:** Some DAWs may not import correctly
   - **Mitigation:** Use standard MIDI format, test with multiple DAWs
2. **Audio Analysis Failures:** Some tracks may be too complex
   - **Mitigation:** Fallback to other source types

---

## Dependencies & Integrations

### Required Dependencies
- **Python Libraries:** See `requirements.txt` in ARCHITECTURE.md
- **External Services:**
  - Anthropic API (primary LLM - Claude 3.5 Sonnet)
  - Google Gemini API (secondary LLM - Gemini 2.5/3)
  - OpenAI API (tertiary/fallback LLM - ChatGPT 5.1)
  - Semantic Scholar API (research)
- **Infrastructure:**
  - PostgreSQL 15+ with pgvector
  - Redis 7.0+ for task queue

### Optional Integrations (Future)
- **Spotify API:** Retrieve artist metadata, audio features
- **MusicBrainz:** Structured music metadata
- **Discogs:** Discography, equipment information
- **Weights & Biases:** ML experiment tracking
- **Sentry:** Error monitoring

---

## Release Plan

### MVP Release (v2.0.0 - Month 3)
- Core research pipeline (4 collectors)
- Multi-provider LLM generation
- PostgreSQL caching
- Max for Live device
- 100+ cached artists (pre-seeded)

### v2.1.0 (Month 4)
- Augmentation feature
- Improved error handling
- Performance optimizations
- 500+ cached artists

### v2.2.0 (Month 5)
- Similarity search
- Web dashboard (beta)
- Usage analytics
- 1,000+ cached artists

### v2.3.0 (Month 6)
- Local LLM support (experimental)
- Style blending (combine artists)
- Community contributions
- 2,000+ cached artists

### v3.0.0 (Month 12)
- VST3 plugin (multi-DAW support)
- Mobile companion app
- Real-time MIDI generation
- 10,000+ cached artists

---

## Open Questions

1. **Pricing Model:** Free tier limits? Subscription pricing?
2. **Commercial Artists:** How to handle very famous artists (licensing concerns)?
3. **User Contributions:** Allow users to submit research sources?
4. **Offline Mode:** Support offline generation with cached profiles?
5. **Multi-Language:** Support non-English artist names and sources?

---

## Appendix A: Competitor Analysis

### Existing Solutions
1. **Splice Beat Maker:** Pre-made loops, no style research, limited customization
2. **XO by XLN Audio:** Pattern sequencer, no AI generation
3. **Addictive Drums 2:** Pre-recorded beats, no on-demand generation
4. **Superior Drummer 3:** MIDI library, no AI, no style research

### Competitive Advantages
- ✓ **Unlimited Artists:** Not limited to pre-trained styles
- ✓ **Research-Backed:** Multi-source style analysis
- ✓ **DAW Integration:** Direct Ableton clip creation
- ✓ **Customizable:** Full parameter control
- ✓ **Scalable:** Cloud-based research and generation

---

## Appendix B: Glossary

- **StyleProfile:** Aggregated data about an artist's drumming style
- **Confidence Score:** 0-1 rating of research data quality
- **LLM:** Large Language Model (GPT-4, Claude, Gemini)
- **Humanization:** Adding micro-variations to timing and velocity
- **Augmentation:** Adding more research sources to existing profile
- **Collector:** Sub-agent that gathers data from a specific source type
- **Orchestrator:** Main coordinator agent managing all sub-agents

---

**Document Status:** ✅ Approved for Implementation
**Next Review Date:** 2025-12-01
**Change Log:**
- 2025-11-17: Initial v2.0 PRD created
