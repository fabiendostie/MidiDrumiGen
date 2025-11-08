# MIDI Operations - Context Document

**Document Type:** Knowledge Context (Domain Knowledge)  
**Purpose:** MIDI protocol, drum mapping, humanization techniques  
**Use Case:** Load when implementing MIDI I/O, export, or processing features

---

## MIDI Fundamentals

### MIDI Protocol Basics

**MIDI** (Musical Instrument Digital Interface) is a technical standard for digital music communication.

**Key Concepts:**
- **Messages**: Note On, Note Off, Control Change, etc.
- **Channels**: 16 channels (0-15 in code, 1-16 in spec)
- **Note Numbers**: 0-127 (Middle C = 60)
- **Velocity**: 0-127 (0 = note off, 1-127 = volume)
- **Timing**: Measured in ticks (480-960 PPQ typical)

**Channel 10 (Index 9)**: Reserved for percussion/drums in General MIDI

---

## General MIDI Drum Mapping

### Standard GM Drum Kit (Channel 10)

| MIDI Note | Drum Name            | Common Name  |
|-----------|---------------------|--------------|
| 35        | Acoustic Bass Drum  | Kick 2       |
| 36        | Bass Drum 1         | Kick 1 ⭐    |
| 37        | Side Stick          | Rimshot      |
| 38        | Acoustic Snare      | Snare ⭐     |
| 39        | Hand Clap           | Clap         |
| 40        | Electric Snare      | Snare 2      |
| 41        | Low Floor Tom       | Tom 1        |
| 42        | Closed Hi-Hat       | Hi-Hat Closed ⭐ |
| 43        | High Floor Tom      | Tom 2        |
| 44        | Pedal Hi-Hat        | Hi-Hat Pedal |
| 45        | Low Tom             | Tom 3        |
| 46        | Open Hi-Hat         | Hi-Hat Open ⭐ |
| 47        | Low-Mid Tom         | Tom 4        |
| 48        | Hi-Mid Tom          | Tom 5        |
| 49        | Crash Cymbal 1      | Crash ⭐     |
| 50        | High Tom            | Tom 6        |
| 51        | Ride Cymbal 1       | Ride ⭐      |
| 52        | Chinese Cymbal      | China        |
| 53        | Ride Bell           | Bell         |
| 54        | Tambourine          | Tambourine   |
| 55        | Splash Cymbal       | Splash       |
| 56        | Cowbell             | Cowbell      |
| 57        | Crash Cymbal 2      | Crash 2      |
| 58        | Vibraslap           | Vibraslap    |
| 59        | Ride Cymbal 2       | Ride 2       |
| 60        | Hi Bongo            | Bongo High   |
| 61        | Low Bongo           | Bongo Low    |
| 62        | Mute Hi Conga       | Conga Mute   |
| 63        | Open Hi Conga       | Conga Open   |
| 64        | Low Conga           | Conga Low    |
| 65        | High Timbale        | Timbale High |
| 66        | Low Timbale         | Timbale Low  |
| 67        | High Agogo          | Agogo High   |
| 68        | Low Agogo           | Agogo Low    |
| 69        | Cabasa              | Cabasa       |
| 70        | Maracas             | Maracas      |
| 71        | Short Whistle       | Whistle Short|
| 72        | Long Whistle        | Whistle Long |
| 73        | Short Guiro         | Guiro Short  |
| 74        | Long Guiro          | Guiro Long   |
| 75        | Claves              | Claves       |
| 76        | Hi Wood Block       | Block High   |
| 77        | Low Wood Block      | Block Low    |
| 78        | Mute Cuica          | Cuica Mute   |
| 79        | Open Cuica          | Cuica Open   |
| 80        | Mute Triangle       | Triangle Mute|
| 81        | Open Triangle       | Triangle Open|

⭐ = Core drums used in most patterns

### Simplified Drum Set (Most Common)

```python
# src/midi/constants.py
CORE_DRUMS = {
    'kick': 36,
    'snare': 38,
    'hihat_closed': 42,
    'hihat_open': 46,
    'crash': 49,
    'ride': 51,
    'tom_low': 43,
    'tom_mid': 47,
    'tom_high': 50,
}

# Reverse mapping
DRUM_NAMES = {v: k for k, v in CORE_DRUMS.items()}
```

---

## MIDI File Structure

### Standard MIDI File (SMF) Format

**Header Chunk:**
- Format type (0, 1, or 2)
- Number of tracks
- Time division (ticks per quarter note)

**Track Chunks:**
- Meta messages (tempo, time signature, track name)
- MIDI events (note on/off, control change)
- Delta times (time since last event)

### Using mido to Create MIDI Files

```python
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

def create_midi_file(
    tempo: int = 120,
    time_signature: tuple = (4, 4),
    ticks_per_beat: int = 480
) -> MidiFile:
    """Create a new MIDI file with proper setup."""
    
    # Initialize file
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Add track name
    track.append(MetaMessage('track_name', name='Drums', time=0))
    
    # Add tempo (microseconds per quarter note)
    tempo_value = mido.bpm2tempo(tempo)
    track.append(MetaMessage('set_tempo', tempo=tempo_value, time=0))
    
    # Add time signature
    numerator, denominator = time_signature
    track.append(MetaMessage(
        'time_signature',
        numerator=numerator,
        denominator=denominator,
        clocks_per_click=24,
        notated_32nd_notes_per_beat=8,
        time=0
    ))
    
    # Set channel to 10 (drums)
    track.append(Message('program_change', program=0, channel=9, time=0))
    
    return mid, track
```

### Adding Notes

```python
def add_note(
    track: MidiTrack,
    note: int,
    velocity: int,
    start_time: int,
    duration: int,
    channel: int = 9  # Drums channel
):
    """Add a note to the track."""
    
    # Note On
    track.append(Message(
        'note_on',
        note=note,
        velocity=velocity,
        time=start_time,
        channel=channel
    ))
    
    # Note Off
    track.append(Message(
        'note_off',
        note=note,
        velocity=0,
        time=duration,
        channel=channel
    ))
```

### Delta Time Calculation

**Delta time** = Time since last event in ticks

```python
def beats_to_ticks(beats: float, ticks_per_beat: int) -> int:
    """Convert beats to MIDI ticks."""
    return int(beats * ticks_per_beat)

# Example: Quarter note at 480 PPQ
quarter_note_ticks = beats_to_ticks(1.0, 480)  # 480
eighth_note_ticks = beats_to_ticks(0.5, 480)   # 240
sixteenth_note_ticks = beats_to_ticks(0.25, 480)  # 120
```

---

## Humanization Techniques

### 1. Timing Variation (Swing & Micro-timing)

**Purpose:** Make patterns feel less robotic, add groove

**Swing Timing:**
```python
def apply_swing(note_time: int, swing_percentage: float, ticks_per_beat: int) -> int:
    """
    Apply swing to offbeat notes.
    
    Args:
        note_time: Original note time in ticks
        swing_percentage: 50-75 (50 = straight, 66 = triplet swing)
        ticks_per_beat: Ticks per quarter note
    
    Returns:
        Adjusted note time
    """
    eighth_note = ticks_per_beat / 2
    
    # Check if note is on an offbeat
    if (note_time / eighth_note) % 2 == 1:
        # Calculate swing offset
        swing_offset = ((swing_percentage - 50) / 50) * (eighth_note / 4)
        return int(note_time + swing_offset)
    
    return note_time

# Examples:
# 50% = straight (no swing)
# 58% = slight swing (humanized)
# 66% = triplet swing (jazz/hip-hop)
# 75% = extreme swing
```

**Micro-timing (Random Offsets):**
```python
import random

def apply_micro_timing(
    note_time: int,
    max_offset_ms: float = 15.0,
    tempo: int = 120,
    ticks_per_beat: int = 480
) -> int:
    """
    Add subtle random timing variations.
    
    Args:
        note_time: Original time in ticks
        max_offset_ms: Maximum offset in milliseconds (±15ms typical)
        tempo: BPM
        ticks_per_beat: PPQ resolution
    
    Returns:
        Humanized time in ticks
    """
    # Convert ms to ticks
    ms_per_tick = (60000 / tempo) / ticks_per_beat
    max_offset_ticks = int(max_offset_ms / ms_per_tick)
    
    # Add random offset
    offset = random.randint(-max_offset_ticks, max_offset_ticks)
    return max(0, note_time + offset)
```

### 2. Velocity Variation

**Purpose:** Add dynamics and human feel

```python
def apply_velocity_variation(
    base_velocity: int,
    variation: float = 0.1,
    min_velocity: int = 1,
    max_velocity: int = 127
) -> int:
    """
    Add random velocity variation.
    
    Args:
        base_velocity: Original velocity (0-127)
        variation: Percentage variation (0.1 = ±10%)
        min_velocity: Minimum allowed velocity
        max_velocity: Maximum allowed velocity
    
    Returns:
        Humanized velocity
    """
    offset = int(base_velocity * variation * random.uniform(-1, 1))
    new_velocity = base_velocity + offset
    return max(min_velocity, min(max_velocity, new_velocity))
```

**Velocity Curves (Accent Patterns):**
```python
def apply_accent_pattern(
    velocities: list[int],
    accent_positions: list[int],
    accent_boost: int = 20
) -> list[int]:
    """
    Apply accent pattern to velocities.
    
    Args:
        velocities: Original velocity values
        accent_positions: Indices to accent (e.g., [0, 2] for beats 1 and 3)
        accent_boost: Velocity increase for accents
    
    Returns:
        Modified velocities
    """
    result = velocities.copy()
    
    for i in accent_positions:
        if i < len(result):
            # Boost accented notes
            result[i] = min(127, result[i] + accent_boost)
            
            # Reduce adjacent notes slightly
            if i > 0:
                result[i-1] = max(1, result[i-1] - 5)
            if i < len(result) - 1:
                result[i+1] = max(1, result[i+1] - 5)
    
    return result
```

### 3. Ghost Notes

**Purpose:** Add subtle, quiet notes between main hits (common in funk/hip-hop)

```python
def add_ghost_notes(
    track: MidiTrack,
    base_notes: list[dict],
    probability: float = 0.3,
    ghost_velocity: int = 30,
    ghost_note: int = 38  # Snare
):
    """
    Add ghost notes between main hits.
    
    Args:
        track: MIDI track to modify
        base_notes: List of main notes with 'time' and 'note' keys
        probability: Chance of adding ghost note (0-1)
        ghost_velocity: Velocity of ghost notes (20-40 typical)
        ghost_note: MIDI note number for ghosts
    """
    for i in range(len(base_notes) - 1):
        if random.random() < probability:
            # Add ghost note between main notes
            curr_time = base_notes[i]['time']
            next_time = base_notes[i + 1]['time']
            ghost_time = (curr_time + next_time) // 2
            
            track.append(Message(
                'note_on',
                note=ghost_note,
                velocity=ghost_velocity,
                time=ghost_time - curr_time,
                channel=9
            ))
            track.append(Message(
                'note_off',
                note=ghost_note,
                velocity=0,
                time=10,  # Very short duration
                channel=9
            ))
```

---

## Pattern Validation

### Musical Validation Rules

```python
def validate_drum_pattern(notes: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate drum pattern for musical correctness.
    
    Args:
        notes: List of note dicts with 'pitch', 'velocity', 'time'
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # 1. Check note range
    for note in notes:
        if not (35 <= note['pitch'] <= 81):
            errors.append(f"Invalid drum note: {note['pitch']}")
    
    # 2. Check velocity range
    for note in notes:
        if not (1 <= note['velocity'] <= 127):
            errors.append(f"Invalid velocity: {note['velocity']}")
    
    # 3. Check density (notes per beat)
    if notes:
        duration = max(n['time'] for n in notes)
        density = len(notes) / (duration / 480)  # Notes per beat
        
        if density < 0.5:
            errors.append("Pattern too sparse (< 0.5 notes/beat)")
        elif density > 16:
            errors.append("Pattern too dense (> 16 notes/beat)")
    
    # 4. Check for simultaneous impossible hits
    # (e.g., closed and open hi-hat at same time)
    EXCLUSIVE_PAIRS = [
        (42, 46),  # Closed/Open hi-hat
        (36, 35),  # Two different kicks
    ]
    
    time_groups = {}
    for note in notes:
        time_groups.setdefault(note['time'], []).append(note['pitch'])
    
    for time, pitches in time_groups.items():
        for p1, p2 in EXCLUSIVE_PAIRS:
            if p1 in pitches and p2 in pitches:
                errors.append(f"Impossible simultaneous notes: {p1}, {p2} at time {time}")
    
    # 5. Check maximum simultaneous hits (realistic limit)
    for time, pitches in time_groups.items():
        if len(pitches) > 4:
            errors.append(f"Too many simultaneous hits ({len(pitches)}) at time {time}")
    
    return len(errors) == 0, errors
```

---

## Complete Export Pipeline

### Full Implementation

```python
# src/midi/export.py
import mido
from pathlib import Path
from typing import List, Dict, Tuple
import random

def export_pattern(
    tokens: list[int],
    tokenizer,
    output_path: Path,
    tempo: int = 120,
    time_signature: Tuple[int, int] = (4, 4),
    humanize: bool = True,
    style_name: str = "Unknown"
) -> Path:
    """
    Export tokenized pattern to MIDI file.
    
    Complete pipeline:
    1. Detokenize tokens to MIDI events
    2. Validate pattern
    3. Apply humanization (if enabled)
    4. Add metadata
    5. Write MIDI file
    """
    
    # 1. Detokenize
    midi_events = tokenizer.decode(tokens)
    
    # 2. Validate
    is_valid, errors = validate_drum_pattern(midi_events)
    if not is_valid:
        raise ValueError(f"Invalid pattern: {errors}")
    
    # 3. Create MIDI file
    mid = MidiFile(ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Add metadata
    track.append(MetaMessage('track_name', name=f'Drums - {style_name}', time=0))
    track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
    
    numerator, denominator = time_signature
    track.append(MetaMessage(
        'time_signature',
        numerator=numerator,
        denominator=denominator,
        clocks_per_click=24,
        notated_32nd_notes_per_beat=8,
        time=0
    ))
    
    # 4. Process and add notes
    last_time = 0
    for event in sorted(midi_events, key=lambda e: e['time']):
        # Get note properties
        note_time = event['time']
        pitch = event['pitch']
        velocity = event['velocity']
        
        # Apply humanization
        if humanize:
            note_time = apply_micro_timing(note_time, max_offset_ms=15.0, tempo=tempo)
            velocity = apply_velocity_variation(velocity, variation=0.1)
        
        # Calculate delta time
        delta_time = note_time - last_time
        
        # Add note on
        track.append(Message(
            'note_on',
            note=pitch,
            velocity=velocity,
            time=delta_time,
            channel=9
        ))
        
        # Add note off (short duration for drums)
        track.append(Message(
            'note_off',
            note=pitch,
            velocity=0,
            time=10,
            channel=9
        ))
        
        last_time = note_time + 10
    
    # 5. Add end of track
    track.append(MetaMessage('end_of_track', time=0))
    
    # 6. Save file
    mid.save(output_path)
    
    return output_path
```

---

## Producer-Specific Styles

### Style Characteristics

```python
# src/midi/styles.py

PRODUCER_STYLES = {
    'j_dilla': {
        'swing': 0.62,  # Signature Dilla swing
        'micro_timing': 0.020,  # More variation
        'ghost_note_prob': 0.4,
        'velocity_variation': 0.15,
        'preferred_tempo': (85, 95),
        'characteristic_notes': [36, 38, 42],  # Kick, snare, closed hat
    },
    'metro_boomin': {
        'swing': 0.52,  # Straighter timing
        'micro_timing': 0.005,  # Tighter quantization
        'ghost_note_prob': 0.1,
        'velocity_variation': 0.08,
        'preferred_tempo': (130, 150),
        'characteristic_notes': [36, 38, 42, 49],  # Add crash
        'trap_rolls': True,  # 32nd note hi-hat rolls
    },
    'questlove': {
        'swing': 0.58,
        'micro_timing': 0.012,
        'ghost_note_prob': 0.5,  # Lots of ghosts
        'velocity_variation': 0.20,  # Very dynamic
        'preferred_tempo': (90, 110),
        'characteristic_notes': [36, 38, 42, 46, 51],  # Full kit
    },
}

def apply_style_humanization(
    notes: list[dict],
    style: str,
    tempo: int
) -> list[dict]:
    """Apply producer-specific humanization."""
    style_params = PRODUCER_STYLES.get(style, {})
    
    humanized = []
    for note in notes:
        # Apply style-specific swing
        if 'swing' in style_params:
            note['time'] = apply_swing(
                note['time'],
                style_params['swing'] * 100,
                480
            )
        
        # Apply micro-timing
        if 'micro_timing' in style_params:
            note['time'] = apply_micro_timing(
                note['time'],
                style_params['micro_timing'] * 1000,
                tempo,
                480
            )
        
        # Velocity variation
        if 'velocity_variation' in style_params:
            note['velocity'] = apply_velocity_variation(
                note['velocity'],
                style_params['velocity_variation']
            )
        
        humanized.append(note)
    
    return humanized
```

---

## Related Documents

- **Project Overview**: `.cursorcontext/01_project_overview.md`
- **Architecture**: `.cursorcontext/02_architecture.md`
- **Dependencies**: `.cursorcontext/03_dependencies.md`
- **ML Pipeline**: `.cursorcontext/05_ml_pipeline.md`
