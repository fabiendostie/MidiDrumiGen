# Epic Technical Specification: MIDI Export & Humanization

Date: 2025-11-19
Author: Fabz
Epic ID: 5
Status: Draft

---

## Overview

Epic 5 implements MIDI file export and humanization for MidiDrumiGen v2.0, transforming LLM-generated JSON into production-ready MIDI files that sound human and natural. The system converts structured JSON output (from Epic 2) into Standard MIDI File Format 0, applies micro-timing variations, velocity humanization, and ghost note insertion based on artist style profiles. The result is MIDI that mimics human drummer imperfections rather than robotic quantization.

This epic serves as the final output stage of the generation pipeline, ensuring MIDI files are compatible with Ableton Live and all major DAWs. It implements three humanization layers (timing, velocity, ghost notes) that are parameterized by StyleProfile quantitative data, making humanization artist-specific rather than generic.

## Objectives and Scope

**In Scope:**
- JSON to MIDI conversion using `mido` library
- MIDI structure validation (note ranges, velocities, timing)
- Micro-timing humanization (±5-20ms jitter based on artist style)
- Velocity humanization (±10-20% variation around mean)
- Ghost note insertion (probability-based, artist-specific)
- Standard MIDI File Format 0 export (single track, GM drums)
- File naming convention: `{artist}_{variation}_YYYYMMDD_HHMMSS.mid`
- MIDI metadata (tempo, time signature, track name)

**Out of Scope:**
- MIDI File Format 1 (multi-track)
- Swing quantization (handled by LLM prompts in Epic 2)
- MIDI effects (reverb, compression) - DAW responsibility
- Real-time MIDI generation (v3.0 feature)
- Pattern variations generator (creates 4-8 patterns) - handled by Epic 2

**Success Criteria:**
- 100% of generated MIDI files load in Ableton Live without errors
- Humanization detectable in timing analysis (std dev > 5ms)
- Velocity variation measurable (std dev > 10)
- Ghost notes present in 80%+ of patterns (when artist profile indicates)
- File size < 50KB per 4-bar pattern
- Conversion time < 500ms per variation

## System Architecture Alignment

This epic aligns with the **MIDI Export & Validation** component in ARCHITECTURE.md Section 2.5 (Generation Layer):

**Referenced Components:**
- MIDI Export & Validation (`src/midi/export.py`)
- Humanization (`src/midi/humanize.py`)
- Validation (`src/midi/validate.py`)

**Architectural Constraints:**
- Use `mido` 1.3.3 for all MIDI I/O (cross-platform)
- Output must be Standard MIDI Format 0 (single track)
- Channel 10 for all drum notes (GM standard)
- Timing resolution: 480 ticks per quarter note (TPQN)
- All operations must be synchronous (no async needed)

**Integration Points:**
- **Input:** MIDI JSON dict from Epic 2 (LLM Generation Engine)
- **Output:** MIDI file paths (List[Path]) to Epic 4 (API Layer) and Epic 6 (Ableton Integration)
- **Dependencies:** `mido`, `numpy` (for humanization algorithms)

## Detailed Design

### Services and Modules

| Module | Responsibility | Input | Output | Owner |
|--------|---------------|-------|--------|-------|
| `src/midi/export.py` | Convert JSON → MIDI file | `midi_json: dict`, `StyleProfile` | `Path` (MIDI file) | MIDI |
| `src/midi/validate.py` | Validate MIDI structure | `midi_json: dict` | `ValidationResult` | MIDI |
| `src/midi/humanize.py` | Apply timing/velocity/ghost note humanization | `MidiTrack`, `StyleProfile` | `MidiTrack` (modified) | MIDI |
| `src/midi/utils.py` | Utility functions (tick conversion, note names) | Various | Various | MIDI |

### Data Models and Contracts

**MIDI JSON Input Schema (from Epic 2):**
```python
{
  "notes": [
    {
      "pitch": 36,       # MIDI note number (35-81 for drums)
      "velocity": 90,    # 1-127
      "time": 0,         # Position in ticks (480 per quarter note)
      "duration": 120    # Note length in ticks
    },
    # ... more notes
  ],
  "tempo": 120,          # BPM
  "time_signature": [4, 4],  # [numerator, denominator]
  "total_bars": 4
}
```

**ValidationResult:**
```python
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]  # Human-readable error messages
    warnings: List[str]  # Non-fatal issues

    def raise_if_invalid(self):
        if not self.valid:
            raise MIDIValidationError('; '.join(self.errors))
```

**HumanizationParams (derived from StyleProfile):**
```python
@dataclass
class HumanizationParams:
    timing_jitter_ms: float  # ±5-20ms
    velocity_variation_percent: float  # ±10-20%
    ghost_note_probability: float  # 0.0-1.0
    ghost_note_velocity_range: tuple[int, int]  # (20, 40)
    swing_percent: float  # 50-67% (not used if LLM already applied)
```

### APIs and Interfaces

**Export Module Public API:**
```python
def export_midi_from_llm(
    midi_json: dict,
    artist_name: str,
    variation_number: int,
    style_profile: StyleProfile,
    output_dir: Path = Path("output"),
    humanize: bool = True
) -> Path:
    """
    Convert LLM JSON to MIDI file with humanization.

    Args:
        midi_json: LLM output (dict with notes, tempo, time_sig)
        artist_name: For file naming
        variation_number: 1-8 (for file naming)
        style_profile: Artist style data for humanization
        output_dir: Directory to save MIDI file
        humanize: Apply humanization (default: True)

    Returns:
        Path to created MIDI file

    Raises:
        MIDIValidationError: If JSON structure invalid
        MIDIExportError: If file writing fails
    """
    # 1. Validate JSON structure
    validate_midi_json(midi_json).raise_if_invalid()

    # 2. Create MIDI file
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    # 3. Add metadata
    track.append(MetaMessage('track_name', name=f"{artist_name} - Variation {variation_number}"))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(midi_json['tempo'])))
    track.append(MetaMessage('time_signature',
                             numerator=midi_json['time_signature'][0],
                             denominator=midi_json['time_signature'][1]))

    # 4. Convert notes to MIDI messages
    for note_data in midi_json['notes']:
        track.append(Message('note_on',
                            channel=9,  # Channel 10 (GM drums)
                            note=note_data['pitch'],
                            velocity=note_data['velocity'],
                            time=note_data['time']))
        track.append(Message('note_off',
                            channel=9,
                            note=note_data['pitch'],
                            velocity=0,
                            time=note_data['time'] + note_data['duration']))

    # 5. Apply humanization
    if humanize:
        humanize_track(track, derive_humanization_params(style_profile))

    # 6. Save to file
    filename = f"{artist_name.lower().replace(' ', '_')}_var{variation_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mid"
    output_path = output_dir / filename
    mid.save(output_path)

    return output_path
```

**Validation Module Public API:**
```python
def validate_midi_json(midi_json: dict) -> ValidationResult:
    """
    Validate MIDI JSON structure and values.

    Checks:
    - Required keys present (notes, tempo, time_signature, total_bars)
    - Notes array not empty
    - Each note has pitch, velocity, time, duration
    - Pitch in valid range (35-81 for drums)
    - Velocity in range (1-127)
    - Time values non-negative
    - Duration > 0
    - Tempo in range (40-300 BPM)
    - Time signature valid
    """
    errors = []
    warnings = []

    # Required keys
    required_keys = ['notes', 'tempo', 'time_signature', 'total_bars']
    for key in required_keys:
        if key not in midi_json:
            errors.append(f"Missing required key: '{key}'")

    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    # Notes validation
    if not midi_json['notes']:
        errors.append("Notes array is empty")

    for i, note in enumerate(midi_json['notes']):
        # Required note fields
        if 'pitch' not in note:
            errors.append(f"Note {i}: Missing 'pitch'")
        elif not (35 <= note['pitch'] <= 81):
            errors.append(f"Note {i}: Pitch {note['pitch']} out of range (35-81)")

        if 'velocity' not in note:
            errors.append(f"Note {i}: Missing 'velocity'")
        elif not (1 <= note['velocity'] <= 127):
            errors.append(f"Note {i}: Velocity {note['velocity']} out of range (1-127)")

        if 'time' not in note:
            errors.append(f"Note {i}: Missing 'time'")
        elif note['time'] < 0:
            errors.append(f"Note {i}: Time cannot be negative")

        if 'duration' not in note:
            errors.append(f"Note {i}: Missing 'duration'")
        elif note['duration'] <= 0:
            errors.append(f"Note {i}: Duration must be positive")

    # Tempo validation
    if not (40 <= midi_json['tempo'] <= 300):
        errors.append(f"Tempo {midi_json['tempo']} out of range (40-300 BPM)")

    # Time signature validation
    ts = midi_json['time_signature']
    if len(ts) != 2:
        errors.append(f"Time signature must be [num, denom], got: {ts}")
    elif ts not in [(4, 4), (3, 4), (5, 4), (6, 8), (7, 8)]:
        warnings.append(f"Unusual time signature: {ts[0]}/{ts[1]}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
```

**Humanization Module Public API:**
```python
def humanize_track(
    track: MidiTrack,
    params: HumanizationParams
) -> None:
    """
    Apply humanization to MIDI track (modifies in-place).

    Applies three humanization layers:
    1. Timing jitter (±5-20ms)
    2. Velocity variation (±10-20%)
    3. Ghost note insertion
    """
    # Layer 1: Timing humanization
    apply_timing_jitter(track, params.timing_jitter_ms)

    # Layer 2: Velocity humanization
    apply_velocity_variation(track, params.velocity_variation_percent)

    # Layer 3: Ghost notes
    add_ghost_notes(track, params)


def apply_timing_jitter(track: MidiTrack, jitter_ms: float) -> None:
    """
    Add random timing variations to note_on messages.

    Jitter is normally distributed: mean=0, std=jitter_ms.
    Converted to ticks (480 TPQN, 120 BPM default).
    """
    import numpy as np

    # Convert ms to ticks (at 120 BPM, 1 quarter = 500ms = 480 ticks)
    ticks_per_ms = 480 / 500  # 0.96 ticks/ms

    for msg in track:
        if msg.type == 'note_on' and msg.velocity > 0:
            # Generate jitter (normal distribution)
            jitter_ticks = int(np.random.normal(0, jitter_ms * ticks_per_ms))

            # Apply jitter (ensure time stays non-negative)
            msg.time = max(0, msg.time + jitter_ticks)


def apply_velocity_variation(track: MidiTrack, variation_percent: float) -> None:
    """
    Vary note velocities around their original values.

    Each velocity is multiplied by random factor:
    factor ~ Normal(1.0, variation_percent/100)
    """
    import numpy as np

    for msg in track:
        if msg.type == 'note_on' and msg.velocity > 0:
            # Generate variation factor
            factor = np.random.normal(1.0, variation_percent / 100)

            # Apply variation (clamp to 1-127)
            new_velocity = int(msg.velocity * factor)
            msg.velocity = max(1, min(127, new_velocity))


def add_ghost_notes(track: MidiTrack, params: HumanizationParams) -> None:
    """
    Insert ghost notes (low-velocity hits) on snare drum.

    Ghost notes are placed:
    - Between existing snare hits
    - With probability defined by params.ghost_note_probability
    - At low velocity (20-40)
    - Slightly off-grid (16th note subdivisions)
    """
    import numpy as np

    snare_notes = [38, 40]  # Snare MIDI notes
    ghost_notes_to_add = []

    # Find all snare hits
    snare_times = [
        msg.time for msg in track
        if msg.type == 'note_on' and msg.note in snare_notes
    ]

    if len(snare_times) < 2:
        return  # Need at least 2 snare hits to insert ghost notes

    # For each pair of consecutive snare hits
    for i in range(len(snare_times) - 1):
        # Decide if ghost note should be added
        if np.random.random() < params.ghost_note_probability:
            # Place ghost note between the two snare hits
            time_gap = snare_times[i + 1] - snare_times[i]
            ghost_time = snare_times[i] + time_gap // 2

            # Random velocity in ghost note range
            ghost_velocity = np.random.randint(*params.ghost_note_velocity_range)

            # Create ghost note (snare, low velocity)
            ghost_notes_to_add.append({
                'type': 'note_on',
                'channel': 9,
                'note': 38,  # Snare
                'velocity': ghost_velocity,
                'time': ghost_time
            })
            ghost_notes_to_add.append({
                'type': 'note_off',
                'channel': 9,
                'note': 38,
                'velocity': 0,
                'time': ghost_time + 60  # Short duration
            })

    # Insert ghost notes into track
    for ghost in ghost_notes_to_add:
        track.append(Message(**ghost))

    # Re-sort track by time (mido handles this internally)
```

### Workflows and Sequencing

**End-to-End MIDI Export Flow:**
```
1. Epic 2 (Generation) → Returns MIDI JSON
   {
     "notes": [...],
     "tempo": 120,
     "time_signature": [4, 4],
     "total_bars": 4
   }

2. validate_midi_json(midi_json)
   → Check all required fields
   → Validate ranges (pitch, velocity, tempo)
   → Return ValidationResult

3. IF validation fails:
   → Raise MIDIValidationError
   → Log error details
   → Return to calling function (Epic 2 HybridCoordinator)

4. Create MIDI file
   → mid = MidiFile(ticks_per_beat=480)
   → track = MidiTrack()
   → Add metadata (tempo, time_sig, track_name)

5. Convert JSON notes → MIDI messages
   FOR each note in midi_json['notes']:
     → Append note_on (channel 9, pitch, velocity, time)
     → Append note_off (channel 9, pitch, 0, time + duration)

6. Apply humanization (if enabled)
   → derive_humanization_params(style_profile)
     → timing_jitter = style_profile.quantitative_params['velocity_std'] * 2
     → velocity_variation = 15.0 (default)
     → ghost_note_prob = style_profile.quantitative_params['ghost_note_prob']

   → apply_timing_jitter(track, timing_jitter_ms)
   → apply_velocity_variation(track, velocity_variation_percent)
   → add_ghost_notes(track, params)

7. Save MIDI file
   → filename = f"{artist}_var{n}_{timestamp}.mid"
   → mid.save(output_dir / filename)
   → Return Path(output_path)

8. Repeat for all variations (4-8)
   → Return List[Path]
```

## Non-Functional Requirements

### Performance

- **Conversion Time:** < 500ms per variation
  - JSON parsing + validation: < 50ms
  - MIDI message creation: < 100ms
  - Humanization: < 200ms
  - File write: < 150ms

- **File Size:** < 50KB per 4-bar pattern
  - Typical: ~5-10KB for 4 bars
  - Max notes: ~1000 per pattern (very dense)

- **Throughput:** Process 10 variations/second (single-threaded)

**Measurement:** Log conversion time per variation, track file sizes

### Security

- **Input Validation:** 100% of invalid JSON rejected
  - Catch malformed JSON (missing keys, wrong types)
  - Catch out-of-range values (pitch, velocity, tempo)
  - Prevent negative time values (could cause errors)

- **File Path Validation:**
  - Sanitize artist names for filenames (remove special chars)
  - Prevent path traversal attacks (validate output_dir)

- **Resource Limits:**
  - Max 10,000 notes per pattern (prevent memory exhaustion)
  - Max file size: 1MB (reject if exceeded)

### Reliability/Availability

- **MIDI Compatibility:** 100% compatibility with Ableton Live 11+
  - Test with Ableton Live 11.3.25
  - Test with Logic Pro, FL Studio, Cubase (spot checks)

- **Error Handling:**
  - Gracefully handle file write errors (disk full, permissions)
  - Retry file write on transient errors (max 3 retries)
  - Log all export errors with stack trace

- **Validation Coverage:** Catch 100% of spec violations
  - Invalid note ranges
  - Invalid velocities
  - Invalid tempos
  - Invalid time signatures

### Observability

- **Logging:**
  - Log each MIDI export (artist, variation, file path, conversion time)
  - Log validation failures (detailed error messages)
  - Log humanization params used

- **Metrics:**
  - `midi_exports_total` (counter)
  - `midi_export_duration_seconds` (histogram)
  - `midi_validation_failures_total` (counter)
  - `midi_file_size_bytes` (histogram)

- **File Metadata:**
  - Embed artist name in track name
  - Embed creation timestamp in filename

## Dependencies and Integrations

**External Dependencies (requirements.txt):**
```python
# MIDI Processing
mido==1.3.3                   # Primary MIDI library (cross-platform)
python-rtmidi==1.5.8          # Real-time MIDI (for future live performance)

# Scientific Computing (for humanization)
numpy==1.26.4

# Utilities
python-dateutil==2.9.0
```

**Version Constraints:**
- mido: >= 1.3.0 (stable API)
- numpy: >= 1.26.0 (for random number generators)

**Integration Points:**

1. **Epic 2 (Generation) → Epic 5:**
   - Input: MIDI JSON dict (list of dicts for variations)
   - Epic 2 calls: `export_midi_from_llm()` for each variation
   - Returns: List[Path] (MIDI file paths)

2. **Epic 5 → Epic 4 (API):**
   - Epic 4 returns file paths in API response
   - Client (Max for Live) downloads files via HTTP

3. **Epic 5 → Epic 6 (Ableton):**
   - Epic 6 reads MIDI files and imports into Ableton clip slots
   - Uses Live API to create clips

4. **Epic 3 (Database):**
   - StyleProfile provides humanization parameters
   - quantitative_params used to derive HumanizationParams

**No External APIs:** All processing is local (no external dependencies)

## Acceptance Criteria (Authoritative)

**AC-1:** validate_midi_json() catches all invalid inputs
- Missing keys (notes, tempo, time_signature, total_bars)
- Empty notes array
- Invalid pitch (< 35 or > 81)
- Invalid velocity (< 1 or > 127)
- Negative time values
- Zero or negative duration
- Invalid tempo (< 40 or > 300 BPM)

**AC-2:** export_midi_from_llm() creates valid MIDI file
- File format: Standard MIDI Format 0
- Channel: 10 (GM drums)
- TPQN: 480 (ticks per quarter note)
- Metadata includes tempo, time signature, track name

**AC-3:** MIDI file loads in Ableton Live without errors
- Test with Ableton Live 11.3.25
- Drag file into session view
- Verify clip plays correctly

**AC-4:** Timing humanization applied (std dev > 5ms)
- Measure timing of note_on messages
- Calculate standard deviation
- Assert std dev > 5ms (detectable jitter)

**AC-5:** Velocity humanization applied (std dev > 10)
- Measure velocities of all notes
- Calculate standard deviation
- Assert std dev > 10 (detectable variation)

**AC-6:** Ghost notes inserted when probability > 0
- Count snare notes before/after humanization
- Verify ghost notes added between existing snare hits
- Ghost note velocities in range (20-40)

**AC-7:** File naming convention followed
- Format: `{artist}_{variation}_YYYYMMDD_HHMMSS.mid`
- Artist name lowercased, spaces replaced with underscores
- Variation number 1-8
- Timestamp in ISO format

**AC-8:** Conversion time < 500ms per variation
- Measure time from JSON input to file saved
- Assert time < 500ms (single-threaded)

**AC-9:** File size < 50KB for 4-bar pattern
- Measure file size of exported MIDI
- Assert size < 50KB (typical: 5-10KB)

**AC-10:** Humanization parameters derived from StyleProfile
- timing_jitter_ms = velocity_std * 2
- ghost_note_prob = quantitative_params['ghost_note_prob']
- Verify params match artist style

## Traceability Mapping

| AC | Spec Section | Component | Test Idea |
|----|-------------|-----------|-----------|
| AC-1 | APIs & Interfaces | `validate_midi_json()` | Unit: Test with invalid JSON, verify all errors caught |
| AC-2 | APIs & Interfaces | `export_midi_from_llm()` | Unit: Export MIDI, verify file structure with mido |
| AC-3 | Non-Functional Req | Full export pipeline | Manual: Load MIDI in Ableton, verify playback |
| AC-4 | Detailed Design | `apply_timing_jitter()` | Unit: Export with humanization, measure std dev |
| AC-5 | Detailed Design | `apply_velocity_variation()` | Unit: Export with humanization, measure std dev |
| AC-6 | Detailed Design | `add_ghost_notes()` | Unit: Count notes before/after, verify ghost notes added |
| AC-7 | APIs & Interfaces | `export_midi_from_llm()` filename logic | Unit: Verify filename format with regex |
| AC-8 | Non-Functional Req | `export_midi_from_llm()` | Performance: Measure conversion time |
| AC-9 | Non-Functional Req | File I/O | Unit: Measure file size after export |
| AC-10 | Detailed Design | `derive_humanization_params()` | Unit: Verify params match StyleProfile data |

## Risks, Assumptions, Open Questions

**Risks:**

1. **MIDI Compatibility Issues (LOW):**
   - **Risk:** Some DAWs may not load MIDI correctly
   - **Mitigation:** Use Standard MIDI Format 0 (universally supported)
   - **Testing:** Test with Ableton, Logic, FL Studio, Cubase

2. **Over-Humanization (MEDIUM):**
   - **Risk:** Too much jitter/variation sounds sloppy
   - **Mitigation:** User feedback loop, adjustable params (future)
   - **Validation:** A/B testing with musicians

3. **File Path Issues (LOW):**
   - **Risk:** Special characters in artist names cause file write errors
   - **Mitigation:** Sanitize filenames (alphanumeric + underscores only)

**Assumptions:**

1. All DAWs support Standard MIDI Format 0 (universal standard)
2. Channel 10 is recognized as GM drums by all DAWs
3. Users prefer humanized MIDI over perfectly quantized
4. 480 TPQN provides sufficient timing resolution

**Open Questions:**

1. **Q:** Should humanization be adjustable by user (slider: 0-100%)?
   - **A:** Out of scope for v2.0, planned for v2.1.0 (user preferences)

2. **Q:** Should we support MIDI File Format 1 (multi-track)?
   - **A:** Out of scope for v2.0, Format 0 is sufficient for drums

3. **Q:** How to handle very dense patterns (>1000 notes)?
   - **A:** Validate and reject (return error to Epic 2)

4. **Q:** Should we embed style profile metadata in MIDI file?
   - **A:** Not supported by Standard MIDI, use track name for artist info

## Test Strategy Summary

**Unit Tests (pytest):**
- `test_validate_midi_json_valid()`: Pass valid JSON, verify ValidationResult.valid=True
- `test_validate_midi_json_missing_keys()`: Pass JSON missing 'notes', verify error
- `test_validate_midi_json_invalid_pitch()`: Pass pitch=200, verify error
- `test_validate_midi_json_invalid_velocity()`: Pass velocity=128, verify error
- `test_export_midi_basic()`: Export simple pattern, verify file created
- `test_export_midi_metadata()`: Verify tempo, time_sig, track_name in MIDI
- `test_timing_jitter_applied()`: Measure std dev of note times, assert > 5ms
- `test_velocity_variation_applied()`: Measure std dev of velocities, assert > 10
- `test_ghost_notes_added()`: Count snare notes before/after, verify increase
- `test_filename_format()`: Verify filename matches pattern
- `test_conversion_time()`: Measure time, assert < 500ms

**Integration Tests:**
- `test_end_to_end_export()`: JSON → MIDI → load in mido → verify structure
- `test_export_multiple_variations()`: Export 4 variations, verify all created
- `test_humanization_with_style_profile()`: Use real StyleProfile, verify params derived

**Manual Tests:**
- Load exported MIDI in Ableton Live 11.3.25
- Load exported MIDI in Logic Pro
- Listen to humanized vs non-humanized patterns (A/B test)

**Performance Tests:**
- `test_export_performance()`: Export 10 variations, measure time
- `test_file_size()`: Export 4-bar pattern, verify size < 50KB

**Coverage Target:** 90%+ for all MIDI modules
