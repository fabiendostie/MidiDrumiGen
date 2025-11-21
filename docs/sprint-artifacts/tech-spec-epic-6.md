# Epic Technical Specification: Ableton Integration

Date: 2025-11-19
Author: Fabz
Epic ID: 6
Status: Draft

---

## Overview

Epic 6 implements the Max for Live device for MidiDrumiGen v2.0, providing seamless Ableton Live integration. The device enables users to generate artist-style drum patterns directly within their DAW workflow without context-switching to external tools. It implements a custom UI (375x600px), JavaScript HTTP bridge to communicate with the FastAPI backend, Live API integration for automatic MIDI clip import, and real-time progress tracking for long-running research tasks.

The device serves as the primary user interface for MidiDrumiGen, handling all user interactions (artist input, parameter selection, generation triggering) and automatically importing generated MIDI files into Ableton's clip slots. It provides visual feedback for all operations (progress bars, status text, confidence indicators) and graceful error handling with actionable user-friendly messages.

## Objectives and Scope

**In Scope:**
- Max for Live device (.amxd) compatible with Ableton Live 11+
- Custom UI layout (375x600px) matching UI.md specification
- JavaScript HTTP bridge using XMLHttpRequest or Fetch API
- Progress polling for async tasks (research, generation)
- MIDI clip import via Live API (automatic clip slot population)
- Error handling with user-friendly messages
- Parameter persistence between sessions (Ableton's parameter storage)
- Device controls: artist input, bars, tempo, time signature, variations
- Buttons: Generate, Augment Research
- Visual indicators: progress bar, status text, confidence rating

**Out of Scope:**
- VST3 plugin for other DAWs (planned for v2.2.0)
- Standalone application (not needed - API server runs independently)
- Real-time MIDI generation during playback (v3.0 feature)
- Multiple artist blending in UI (v2.3.0 feature)
- Custom color themes (future enhancement)

**Success Criteria:**
- Device loads in Ableton Live 11+ without errors
- All API calls complete successfully (research, generation, task polling)
- Generated MIDI clips appear in clip slots automatically
- Progress bar updates in real-time during research (< 5 second lag)
- Error messages are clear and actionable (no technical jargon)
- Device state persists between Ableton sessions
- UI is usable at 100% zoom (no scaling issues)

## System Architecture Alignment

This epic aligns with the **User Layer** defined in ARCHITECTURE.md Section 1:

**Referenced Components:**
- Max for Live Device (`ableton/MidiDrumiGen.amxd`)
- JavaScript HTTP Bridge (`ableton/js/bridge.js`)
- Live API integration (built into Max)

**Architectural Constraints:**
- Max 8.5+ required (for JavaScript object)
- Ableton Live 11+ required (for Live API 12.0)
- HTTP communication to localhost:8000 (FastAPI server)
- All API calls must be asynchronous (non-blocking UI)
- MIDI clip import uses Live API's `create_midi_clip()` method

**Integration Points:**
- **Epic 4 (API Layer):** All HTTP requests go through FastAPI endpoints
  - POST /api/v1/research
  - GET /api/v1/research/{artist}
  - POST /api/v1/generate
  - GET /api/v1/task/{task_id}
  - POST /api/v1/augment/{artist}

- **Epic 5 (MIDI Export):** Downloads MIDI files and imports into Ableton
  - MIDI files stored in local directory (output/)
  - Live API reads files and creates clips

## Detailed Design

### Services and Modules

| Module | Responsibility | Input | Output | Owner |
|--------|---------------|-------|--------|-------|
| `MidiDrumiGen.amxd` | Main Max for Live device (UI + logic) | User interactions | HTTP requests, Live API calls | M4L |
| `js/bridge.js` | JavaScript HTTP bridge to FastAPI | API endpoint, params | JSON responses | M4L |
| `max_patches/ui.maxpat` | UI layout (bpatcher or presentation mode) | N/A | Visual elements | M4L |
| `max_patches/live_api.maxpat` | Live API integration (clip import) | MIDI file paths | Clip slots populated | M4L |

### Data Models and Contracts

**Device Parameters (stored in Ableton):**
```javascript
{
  "artist": "",                // Text input (default: empty)
  "bars": 4,                   // Dropdown: 1-16 (default: 4)
  "tempo": 120,                // Number box: 40-300 (default: 120)
  "time_signature": "4/4",     // Dropdown: "4/4", "3/4", "5/4", "6/8", "7/8" (default: "4/4")
  "variations": 4,             // Dropdown: 1-8 (default: 4)
  "provider": "auto",          // Dropdown: "auto", "anthropic", "google", "openai" (default: "auto")
  "humanize": true             // Toggle: on/off (default: on)
}
```

**Device State (runtime, not persisted):**
```javascript
{
  "status": "Ready",           // Status text (Ready, Researching, Generating, Error, etc.)
  "progress": 0,               // Progress bar value (0-100)
  "confidence": 0.0,           // Confidence score (0.0-1.0)
  "cached": false,             // Is artist cached?
  "task_id": null,             // Current Celery task ID (for polling)
  "polling_interval": null     // Polling timer reference
}
```

**HTTP Request/Response Models:**

See Epic 4 spec for detailed API contracts. Device uses these endpoints:

- POST /api/v1/research → `{"task_id": "...", "status": "researching", "estimated_time_minutes": 15}`
- GET /api/v1/research/{artist} → `{"exists": true, "confidence": 0.85, ...}`
- POST /api/v1/generate → `{"status": "success", "midi_files": [...], ...}`
- GET /api/v1/task/{task_id} → `{"status": "completed", "progress": 100, "result": {...}}`

### APIs and Interfaces

**JavaScript HTTP Bridge (bridge.js):**

```javascript
// Initialize HTTP bridge
const API_BASE_URL = "http://localhost:8000/api/v1";

// Helper function: Make HTTP request
async function makeRequest(method, endpoint, body = null) {
    const url = `${API_BASE_URL}${endpoint}`;
    const options = {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        }
    };

    if (body) {
        options.body = JSON.stringify(body);
    }

    try {
        const response = await fetch(url, options);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || `HTTP ${response.status}`);
        }

        return data;
    } catch (error) {
        post(`ERROR: ${error.message}\n`);
        throw error;
    }
}

// API Functions
async function checkArtistCached(artist) {
    return await makeRequest('GET', `/research/${encodeURIComponent(artist)}`);
}

async function triggerResearch(artist, depth = "full") {
    return await makeRequest('POST', '/research', {artist, depth});
}

async function generatePattern(params) {
    return await makeRequest('POST', '/generate', params);
}

async function checkTaskStatus(task_id) {
    return await makeRequest('GET', `/task/${task_id}`);
}

async function augmentResearch(artist) {
    return await makeRequest('POST', `/augment/${encodeURIComponent(artist)}`);
}

// Polling function (called every 2 seconds during research)
function pollTaskStatus(task_id, callback) {
    const interval = setInterval(async () => {
        try {
            const status = await checkTaskStatus(task_id);

            // Update UI
            outlet(0, "progress", status.progress);
            outlet(0, "status", status.current_step || "Processing...");

            // Check if complete
            if (status.status === "completed") {
                clearInterval(interval);
                callback(null, status.result);
            } else if (status.status === "failed") {
                clearInterval(interval);
                callback(new Error(status.error), null);
            }
        } catch (error) {
            clearInterval(interval);
            callback(error, null);
        }
    }, 2000);  // Poll every 2 seconds

    return interval;  // Return interval ID for cleanup
}
```

**Max Patcher Logic (main device flow):**

```max
// Main message flow in Max
[inlet] → [route generate augment check_artist]
          ↓
      [route generate]
          ↓
      [js bridge.js]  // Call generatePattern()
          ↓
      [unpack]  // Parse JSON response
          ↓
      [js live_api.js]  // Import MIDI clips
          ↓
      [outlet] → [live.path]  // Navigate to clip slots
          ↓
      [live.object]  // Create clips via Live API
```

**Live API Integration (clip import):**

```javascript
// Live API: Import MIDI clips into session
function importMIDIClips(midiFilePaths, trackIndex = 0) {
    const liveAPI = new LiveAPI();

    // Navigate to track
    liveAPI.path = `live_set tracks ${trackIndex}`;

    // Get clip slots
    const clipSlots = liveAPI.get('clip_slots');
    const numSlots = clipSlots.length;

    // Import each MIDI file into next available slot
    midiFilePaths.forEach((filePath, index) => {
        if (index < numSlots) {
            // Find empty slot
            liveAPI.path = `live_set tracks ${trackIndex} clip_slots ${index}`;

            // Check if slot is empty
            const hasClip = liveAPI.get('has_clip')[0];

            if (!hasClip) {
                // Create clip from MIDI file
                liveAPI.call('create_clip', 4);  // 4 bars (adjust based on params)

                // Set clip name
                liveAPI.path = `live_set tracks ${trackIndex} clip_slots ${index} clip`;
                liveAPI.set('name', `${artistName} - Variation ${index + 1}`);

                // Import MIDI data (requires file path)
                // Note: Live API doesn't support direct file import
                // Alternative: Parse MIDI and set notes programmatically
                importMIDIFile(liveAPI, filePath);
            }
        }
    });

    post("✓ Imported " + midiFilePaths.length + " clips\n");
}

// Helper: Parse MIDI file and set notes via Live API
function importMIDIFile(liveAPI, filePath) {
    // Read MIDI file using mido (Python)
    // Convert to Live API note format
    // Call liveAPI.call('set_notes', notes)
    // This is complex - may require Python bridge or Max's mxj
}
```

### Workflows and Sequencing

**Workflow 1: Generate Pattern for Cached Artist**
```
1. User enters artist name ("John Bonham")
   → Click "Generate" button

2. Device → GET /api/v1/research/John%20Bonham
   → Check if cached

3. API Response: {"exists": true, "confidence": 0.85}
   → Update UI: Show confidence indicator (●●●●◐)
   → Enable "Generate" button

4. User → Click "Generate"

5. Device → POST /api/v1/generate
   Body: {
     "artist": "John Bonham",
     "bars": 4,
     "tempo": 120,
     "time_signature": [4, 4],
     "variations": 4
   }

6. Show progress: "Generating patterns..."
   → Disable buttons during generation

7. API Response (2 minutes later):
   {
     "status": "success",
     "midi_files": [
       "output/john_bonham_var1_20251119_143022.mid",
       "output/john_bonham_var2_20251119_143022.mid",
       "output/john_bonham_var3_20251119_143022.mid",
       "output/john_bonham_var4_20251119_143022.mid"
     ],
     "generation_time_ms": 87432,
     "provider_used": "anthropic"
   }

8. Device → Import MIDI clips
   → Call Live API to create clips in slots
   → Set clip names: "John Bonham - Variation 1", etc.

9. Update UI:
   → Status: "✓ Complete!"
   → Progress: 100%
   → Re-enable buttons

10. User sees MIDI clips in Ableton session
```

**Workflow 2: Research New Artist + Generate**
```
1. User enters artist name ("Questlove")
   → Click "Generate" button

2. Device → GET /api/v1/research/Questlove
   → Check if cached

3. API Response: {"exists": false}
   → Show prompt: "Artist not found. Research first?"
   → User clicks "Yes, research"

4. Device → POST /api/v1/research
   Body: {"artist": "Questlove", "depth": "full"}

5. API Response: {"task_id": "uuid-123", "status": "researching", "estimated_time_minutes": 15}

6. Start polling:
   → Every 2 seconds → GET /api/v1/task/uuid-123
   → Update progress bar (0 → 25 → 50 → 75 → 100)
   → Update status text:
     - "Searching papers..."
     - "Scraping articles..."
     - "Analyzing audio..."
     - "Building profile..."

7. Task complete:
   → Status: "completed"
   → Result: {"artist_id": 456, "confidence": 0.78}

8. Update UI:
   → Status: "Research complete! Generating patterns..."
   → Automatically trigger generation (same as Workflow 1, step 5)

9. ... (continue with generation flow)
```

**Workflow 3: Augment Existing Research**
```
1. User enters cached artist name
   → Click "Augment Research" button

2. Device → POST /api/v1/augment/John%20Bonham

3. API Response: {"task_id": "uuid-456", "status": "augmenting", "estimated_time_minutes": 10}

4. Start polling (same as Workflow 2, step 6)

5. Task complete:
   → Update confidence indicator (e.g., 0.85 → 0.91)
   → Status: "✓ Research augmented!"
```

## Non-Functional Requirements

### Performance

- **UI Responsiveness:** < 100ms for all button clicks
  - Generate button → HTTP request fires immediately
  - No UI freezing during API calls (async operations)

- **Progress Update Frequency:** Poll every 2 seconds
  - Balance between real-time updates and server load
  - Progress bar updates smoothly (no jumps)

- **MIDI Clip Import Time:** < 5 seconds for 4 clips
  - Live API clip creation: ~1 second per clip
  - Total: ~4 seconds for 4 variations

**Measurement:** User perception testing, timer logs in Max console

### Security

- **API URL Hardcoded:** localhost:8000 only (development)
  - Production: Allow user to configure API URL (future)

- **Input Validation:** Client-side validation before API call
  - Artist name: 1-100 chars, alphanumeric + spaces
  - Tempo: 40-300 BPM
  - Bars: 1-16

- **Error Handling:** Never expose technical errors to user
  - HTTP 500 → "Server error. Please try again."
  - Network timeout → "Connection error. Check if API server is running."

### Reliability/Availability

- **Graceful Degradation:**
  - If API server down → Show "Server unreachable" message
  - If research fails → Offer retry or skip
  - If generation fails → Show error, keep previous state

- **State Persistence:**
  - All parameters persist between Ableton sessions
  - Last artist name remembered
  - User doesn't lose input on device reload

- **Error Recovery:**
  - Network timeout: Auto-retry once after 5 seconds
  - Task polling failure: Stop polling, show error
  - Clip import failure: Log error, don't crash Ableton

### Observability

- **Logging:** Max console logs for debugging
  - Log all API requests (method, endpoint, params)
  - Log all API responses (status, body)
  - Log errors with stack trace

- **User Feedback:** All operations have visual feedback
  - Progress bar for long operations
  - Status text updates in real-time
  - Error messages in red text

- **Metrics:** (Future enhancement)
  - Track button clicks (telemetry)
  - Track generation success rate
  - Track average research time

## Dependencies and Integrations

**External Dependencies:**
- **Max for Live:** Max 8.5+ (includes JavaScript object)
- **Ableton Live:** 11+ (includes Live API 12.0)
- **Node.js:** Not required (JavaScript runs in Max's JS engine)

**Max Packages/Externals:**
- `js` object (JavaScript engine)
- `live.path`, `live.object` (Live API navigation)
- `bpatcher` or `presentation` (UI layout)
- `umenu`, `textbutton`, `live.slider` (UI controls)

**Integration Points:**

1. **Epic 4 (API Layer):**
   - All HTTP requests via JavaScript Fetch API
   - Endpoints: /research, /generate, /task/{id}, /augment

2. **Epic 5 (MIDI Export):**
   - Device downloads MIDI files from local filesystem
   - Files stored in `output/` directory

3. **Ableton Live API:**
   - Clip creation: `create_clip(length_in_bars)`
   - Clip naming: `set('name', value)`
   - Note insertion: `set_notes(notes_array)` or file import

## Acceptance Criteria (Authoritative)

**AC-1:** Device loads in Ableton Live 11+ without errors
- Drag `.amxd` file to MIDI track
- Device appears in device chain
- UI renders correctly at 375x600px

**AC-2:** UI matches specification in UI.md
- Artist input field (text)
- Bars dropdown (1-16)
- Tempo number box (40-300)
- Time signature dropdown (4/4, 3/4, 5/4, 6/8, 7/8)
- Variations dropdown (1-8)
- Generate button
- Augment button
- Progress bar
- Status text
- Confidence indicator (●●●●◐ style)

**AC-3:** Generate button triggers POST /api/v1/generate
- Verify HTTP request sent with correct params
- Verify response handled (MIDI files imported)

**AC-4:** Augment button triggers POST /api/v1/augment
- Verify HTTP request sent
- Verify polling started
- Verify progress bar updates

**AC-5:** Progress polling updates UI every 2 seconds
- Start research task
- Verify GET /api/v1/task/{id} called every 2 seconds
- Verify progress bar updates (0 → 100)
- Verify status text updates

**AC-6:** MIDI clips imported into Ableton automatically
- Generate 4 variations
- Verify 4 clips appear in clip slots
- Verify clip names: "{Artist} - Variation 1", etc.

**AC-7:** Error messages are user-friendly
- Simulate API server down → "Server unreachable"
- Simulate artist not found → "Artist not researched. Research first?"
- Simulate generation failure → "Generation failed. Try again."

**AC-8:** Parameters persist between sessions
- Set artist="John Bonham", bars=8, tempo=95
- Save and close Ableton project
- Reopen project → Verify parameters restored

**AC-9:** Confidence indicator displays correctly
- Load cached artist with confidence=0.85
- Verify indicator shows 4.25 filled dots (●●●●◐)

**AC-10:** Device doesn't crash Ableton
- Perform 10 consecutive generations
- Verify Ableton remains stable

## Traceability Mapping

| AC | Spec Section | Component | Test Idea |
|----|-------------|-----------|-----------|
| AC-1 | Overall | `MidiDrumiGen.amxd` | Manual: Load in Ableton, verify no errors |
| AC-2 | Detailed Design | UI layout | Manual: Visual inspection against UI.md |
| AC-3 | APIs & Interfaces | `bridge.js` generatePattern() | Unit: Mock API, verify HTTP request |
| AC-4 | APIs & Interfaces | `bridge.js` augmentResearch() | Unit: Mock API, verify HTTP request |
| AC-5 | Workflows | `bridge.js` pollTaskStatus() | Integration: Start task, verify polling |
| AC-6 | Detailed Design | `live_api.js` importMIDIClips() | Integration: Generate, verify clips in Ableton |
| AC-7 | Non-Functional Req | Error handling | Integration: Simulate errors, verify messages |
| AC-8 | Non-Functional Req | Ableton parameter storage | Manual: Save/reload project, verify params |
| AC-9 | Detailed Design | UI confidence indicator | Manual: Load artist, verify dots match score |
| AC-10 | Non-Functional Req | Overall stability | Stress test: 10 generations, verify no crash |

## Risks, Assumptions, Open Questions

**Risks:**

1. **Live API Limitations (HIGH):**
   - **Risk:** Live API may not support direct MIDI file import
   - **Mitigation:** Parse MIDI and set notes programmatically using `set_notes()`
   - **Workaround:** Prompt user to drag files manually (not ideal)

2. **JavaScript Engine Compatibility (MEDIUM):**
   - **Risk:** Max's JS engine may not support modern ES6+ features
   - **Mitigation:** Transpile JavaScript to ES5 if needed (Babel)
   - **Testing:** Test on multiple Max versions (8.5, 8.6)

3. **Ableton Version Compatibility (LOW):**
   - **Risk:** Device may not work on older Ableton versions
   - **Mitigation:** Require Ableton 11+ (clearly documented)
   - **Testing:** Test on Live 11.0, 11.1, 11.2, 11.3

**Assumptions:**

1. API server runs on localhost:8000 (same machine as Ableton)
2. Users can start/stop API server independently (Python script)
3. Max's JavaScript object supports Fetch API or XMLHttpRequest
4. Live API supports clip creation and naming (verified in docs)

**Open Questions:**

1. **Q:** Should device auto-start API server on load?
   - **A:** No - too complex, requires Python integration. User starts manually.

2. **Q:** How to handle MIDI file paths on Windows vs macOS?
   - **A:** Use relative paths from output/ directory (cross-platform)

3. **Q:** Should device support drag-and-drop artist names from Ableton browser?
   - **A:** Out of scope for v2.0 (complex UI integration)

4. **Q:** Should we implement undo/redo for clip imports?
   - **A:** Out of scope - users can delete clips manually in Ableton

## Test Strategy Summary

**Unit Tests (Jest - for JavaScript):**
- `test_make_request()`: Mock Fetch API, verify HTTP calls
- `test_poll_task_status()`: Mock polling, verify interval timing
- `test_error_handling()`: Simulate network errors, verify error messages

**Integration Tests (Manual - in Ableton):**
- `test_generate_flow()`: End-to-end generation (cached artist)
- `test_research_flow()`: End-to-end research + generation (new artist)
- `test_augment_flow()`: Augment existing artist, verify confidence update
- `test_clip_import()`: Verify clips appear in session view
- `test_parameter_persistence()`: Save/reload project, verify params

**Compatibility Tests:**
- Test on Ableton Live 11.0, 11.1, 11.2, 11.3
- Test on Max 8.5, 8.6
- Test on Windows 10, Windows 11, macOS 11+

**Stress Tests:**
- 10 consecutive generations (verify no memory leaks)
- 50 rapid button clicks (verify no UI freezing)
- Long research task (20 minutes) - verify polling doesn't timeout

**Coverage Target:** 80%+ for JavaScript bridge code (unit tests)
