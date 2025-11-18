# MIDI Operations - MidiDrumiGen v2.0

**Version:** 2.0.0  
**Last Updated:** 2025-11-17  
**Library:** mido 1.3.2

---

## Overview

MIDI operations in v2.0 handle:
1. **LLM JSON → MIDI Conversion** (NEW)
2. **Humanization** (KEPT)
3. **Style Transfer** (KEPT)
4. **Validation** (ENHANCED)
5. **Export** (ENHANCED)

**Key Change:** v2.0 generates MIDI from LLM JSON output instead of PyTorch model tokens.

---

## Core MIDI Module

Location: `src/midi/`

### Files
- `export.py` - JSON to MIDI conversion, file export
- `humanize.py` - Timing and velocity humanization
- `style_transfer.py` - Apply style characteristics
- `validate.py` - MIDI structure validation
- `constants.py` - GM drum mapping, MIDI constants
- `io.py` - MIDI file I/O utilities

---

## 1. LLM JSON to MIDI Conversion (NEW)

### Input Format (from LLM)
```python
{
  "notes": [
    {
      "pitch": 36,      # MIDI note number (35-81 for drums)
      "velocity": 90,   # 1-127
      "time": 0,        # Position in ticks (480 per quarter note)
      "duration": 120   # Note length in ticks
    },
    {
      "pitch": 42,
      "velocity": 70,
      "time": 240,
      "duration": 120
    }
  ],
  "tempo": 120,
  "time_signature": [4, 4],
  "total_bars": 4
}
```

### Conversion Function
```python
# src/midi/export.py

from mido import MidiFile, MidiTrack, Message, MetaMessage
from pathlib import Path
from typing import Dict, List

def json_to_midi(
    midi_data: Dict,
    output_path: Path,
    humanize: bool = True,
    style_params: Dict = None
) -> Path:
    """
    Convert LLM JSON output to MIDI file.
    
    Args:
        midi_data: JSON from LLM with notes array
        output_path: Where to save MIDI file
        humanize: Apply timing/velocity variations
        style_params: Optional style-specific humanization
    
    Returns:
        Path to created MIDI file
    """
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    tempo_bpm = midi_data.get('tempo', 120)
    tempo_us = mido.bpm2tempo(tempo_bpm)
    track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))
    
    # Set time signature
    ts = midi_data.get('time_signature', [4, 4])
    track.append(MetaMessage(
        'time_signature',
        numerator=ts[0],
        denominator=ts[1],
        time=0
    ))
    
    # Sort notes by time
    notes = sorted(midi_data['notes'], key=lambda n: n['time'])
    
    # Convert to MIDI messages
    current_time = 0
    note_offs = []  # Track note_off messages
    
    for note in notes:
        # Calculate delta time from last event
        delta_time = note['time'] - current_time
        
        # Apply humanization if enabled
        if humanize:
            delta_time, velocity = apply_humanization(
                delta_time,
                note['velocity'],
                style_params
            )
        else:
            velocity = note['velocity']
        
        # Note on
        track.append(Message(
            'note_on',
            note=note['pitch'],
            velocity=velocity,
            time=delta_time
        ))
        
        # Schedule note off
        note_offs.append({
            'time': note['time'] + note['duration'],
            'pitch': note['pitch']
        })
        
        current_time = note['time']
    
    # Add note off messages
    note_offs.sort(key=lambda n: n['time'])
    for note_off in note_offs:
        delta_time = note_off['time'] - current_time
        track.append(Message(
            'note_off',
            note=note_off['pitch'],
            velocity=0,
            time=delta_time
        ))
        current_time = note_off['time']
    
    # Save file
    mid.save(output_path)
    return output_path
```

---

## 2. Humanization

### Purpose
Add micro-variations to make MIDI sound less robotic and more like a human drummer.

### Parameters
```python
@dataclass
class HumanizationParams:
    swing: float = 50.0               # 50-67% (50=straight, 62=swing)
    micro_timing_ms: float = 10.0     # ±timing offset in milliseconds
    velocity_variation: float = 0.12   # ±12% velocity variation
    ghost_note_prob: float = 0.0       # Probability to add ghost notes
```

### Implementation
```python
# src/midi/humanize.py

import random
import numpy as np

def apply_humanization(
    delta_time: int,
    velocity: int,
    params: HumanizationParams
) -> tuple[int, int]:
    """
    Apply humanization to timing and velocity.
    
    Args:
        delta_time: Original timing in ticks
        velocity: Original velocity (1-127)
        params: Humanization parameters
    
    Returns:
        (humanized_time, humanized_velocity)
    """
    # Apply swing (affects timing on off-beats)
    if params.swing != 50.0:
        delta_time = apply_swing(delta_time, params.swing)
    
    # Apply micro-timing variations
    variance_ticks = int((params.micro_timing_ms / 1000) * 480)  # 480 tpqn
    timing_offset = random.randint(-variance_ticks, variance_ticks)
    delta_time = max(0, delta_time + timing_offset)
    
    # Apply velocity variations
    velocity_offset = int(velocity * random.uniform(
        -params.velocity_variation,
        params.velocity_variation
    ))
    humanized_velocity = np.clip(velocity + velocity_offset, 1, 127)
    
    return delta_time, int(humanized_velocity)


def apply_swing(delta_time: int, swing_percent: float) -> int:
    """
    Apply swing feel to timing.
    
    Swing delays the off-beat notes (2nd, 4th, 6th, 8th of 8th notes).
    
    Args:
        delta_time: Original timing
        swing_percent: 50-67 (50=straight, 62=typical swing)
    
    Returns:
        Adjusted timing
    """
    # Detect if this is an off-beat (every other 8th note)
    eighth_note_ticks = 240  # At 480 tpqn
    position_in_beat = delta_time % eighth_note_ticks
    
    if position_in_beat == 0:  # This is an off-beat
        # Calculate swing delay
        swing_ratio = swing_percent / 50.0  # 1.0 = straight, 1.24 = 62% swing
        swing_delay = int(eighth_note_ticks * (swing_ratio - 1.0))
        return delta_time + swing_delay
    
    return delta_time


def add_ghost_notes(
    notes: List[Dict],
    probability: float = 0.3
) -> List[Dict]:
    """
    Add ghost notes (soft hi-hat or snare hits) to pattern.
    
    Args:
        notes: Existing note list
        probability: Chance to add ghost note (0.0-1.0)
    
    Returns:
        Notes with ghost notes added
    """
    ghost_notes = []
    
    # Find spaces between notes where ghost notes can go
    for i in range(len(notes) - 1):
        if random.random() < probability:
            # Add ghost note between current and next note
            time_gap = notes[i+1]['time'] - notes[i]['time']
            
            if time_gap > 240:  # Only if gap is > 8th note
                ghost_time = notes[i]['time'] + (time_gap // 2)
                ghost_notes.append({
                    'pitch': 42,  # Closed hi-hat
                    'velocity': random.randint(30, 50),  # Soft
                    'time': ghost_time,
                    'duration': 60
                })
    
    # Merge and sort
    all_notes = notes + ghost_notes
    return sorted(all_notes, key=lambda n: n['time'])
```

---

## 3. Style Transfer

Apply style-specific characteristics from StyleProfile.

```python
# src/midi/style_transfer.py

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class StyleCharacteristics:
    """Style characteristics from StyleProfile."""
    tempo_range: tuple[int, int]  # (min, max) BPM
    swing_percent: float           # 50-67%
    ghost_note_prob: float         # 0.0-1.0
    velocity_mean: int             # 40-120
    velocity_std: int              # 5-30
    syncopation_level: float       # 0.0-1.0


def apply_style_transfer(
    notes: List[Dict],
    style: StyleCharacteristics
) -> List[Dict]:
    """
    Apply style characteristics to MIDI notes.
    
    This modifies the notes to match the artist's documented style.
    
    Args:
        notes: Original notes from LLM
        style: Style characteristics from StyleProfile
    
    Returns:
        Modified notes with style applied
    """
    # 1. Adjust velocities to match artist's typical range
    notes = adjust_velocity_distribution(notes, style.velocity_mean, style.velocity_std)
    
    # 2. Add ghost notes if characteristic of artist
    if style.ghost_note_prob > 0.2:
        notes = add_ghost_notes(notes, style.ghost_note_prob)
    
    # 3. Apply syncopation patterns
    if style.syncopation_level > 0.5:
        notes = add_syncopation(notes, style.syncopation_level)
    
    return notes


def adjust_velocity_distribution(
    notes: List[Dict],
    target_mean: int,
    target_std: int
) -> List[Dict]:
    """
    Adjust velocity distribution to match artist's typical dynamics.
    """
    import numpy as np
    
    # Calculate current distribution
    velocities = [n['velocity'] for n in notes]
    current_mean = np.mean(velocities)
    current_std = np.std(velocities)
    
    # Adjust each velocity
    for note in notes:
        # Normalize to 0-1
        normalized = (note['velocity'] - current_mean) / current_std
        # Scale to target distribution
        new_velocity = int(normalized * target_std + target_mean)
        # Clip to valid range
        note['velocity'] = np.clip(new_velocity, 1, 127)
    
    return notes
```

---

## 4. Validation

Ensure MIDI output is valid and playable.

```python
# src/midi/validate.py

from typing import Dict, List, Optional

class MIDIValidationError(Exception):
    """Raised when MIDI validation fails."""
    pass


def validate_midi_json(midi_data: Dict) -> Optional[str]:
    """
    Validate LLM JSON output before conversion to MIDI.
    
    Args:
        midi_data: JSON from LLM
    
    Returns:
        None if valid, error message if invalid
    """
    # Check required fields
    if 'notes' not in midi_data:
        return "Missing 'notes' array"
    
    if not isinstance(midi_data['notes'], list):
        return "'notes' must be an array"
    
    if len(midi_data['notes']) == 0:
        return "Empty 'notes' array"
    
    # Validate each note
    for i, note in enumerate(midi_data['notes']):
        # Check required fields
        required_fields = ['pitch', 'velocity', 'time', 'duration']
        for field in required_fields:
            if field not in note:
                return f"Note {i} missing '{field}'"
        
        # Validate pitch (35-81 for GM drums)
        if not (35 <= note['pitch'] <= 81):
            return f"Note {i} pitch {note['pitch']} out of range (35-81)"
        
        # Validate velocity (1-127)
        if not (1 <= note['velocity'] <= 127):
            return f"Note {i} velocity {note['velocity']} out of range (1-127)"
        
        # Validate timing (non-negative)
        if note['time'] < 0:
            return f"Note {i} time {note['time']} is negative"
        
        # Validate duration (positive)
        if note['duration'] <= 0:
            return f"Note {i} duration {note['duration']} is non-positive"
    
    # Validate tempo
    if 'tempo' in midi_data:
        tempo = midi_data['tempo']
        if not (40 <= tempo <= 300):
            return f"Tempo {tempo} out of range (40-300 BPM)"
    
    # Validate time signature
    if 'time_signature' in midi_data:
        ts = midi_data['time_signature']
        if not isinstance(ts, list) or len(ts) != 2:
            return "time_signature must be [numerator, denominator]"
        if ts[0] not in [3, 4, 5, 6, 7]:
            return f"Invalid time signature numerator: {ts[0]}"
        if ts[1] not in [4, 8]:
            return f"Invalid time signature denominator: {ts[1]}"
    
    return None  # Valid


def validate_midi_file(file_path: Path) -> bool:
    """
    Validate MIDI file can be read and is properly formatted.
    
    Args:
        file_path: Path to MIDI file
    
    Returns:
        True if valid, raises exception if invalid
    """
    try:
        mid = MidiFile(file_path)
        
        # Check has at least one track
        if len(mid.tracks) == 0:
            raise MIDIValidationError("MIDI file has no tracks")
        
        # Check for note events
        has_notes = False
        for track in mid.tracks:
            for msg in track:
                if msg.type in ['note_on', 'note_off']:
                    has_notes = True
                    break
        
        if not has_notes:
            raise MIDIValidationError("MIDI file has no note events")
        
        return True
        
    except Exception as e:
        raise MIDIValidationError(f"Invalid MIDI file: {e}")
```

---

## 5. GM Drum Mapping

Standard General MIDI drum note assignments.

```python
# src/midi/constants.py

# GM Drum Map (MIDI Channel 10)
GM_DRUM_MAP = {
    # Kick Drums
    35: "Acoustic Bass Drum",
    36: "Bass Drum 1",
    
    # Snares
    38: "Acoustic Snare",
    40: "Electric Snare",
    37: "Side Stick",
    
    # Hi-Hats
    42: "Closed Hi-Hat",
    44: "Pedal Hi-Hat",
    46: "Open Hi-Hat",
    
    # Cymbals
    49: "Crash Cymbal 1",
    51: "Ride Cymbal 1",
    52: "Chinese Cymbal",
    53: "Ride Bell",
    55: "Splash Cymbal",
    57: "Crash Cymbal 2",
    59: "Ride Cymbal 2",
    
    # Toms
    41: "Low Floor Tom",
    43: "High Floor Tom",
    45: "Low Tom",
    47: "Low-Mid Tom",
    48: "Hi-Mid Tom",
    50: "High Tom",
    
    # Percussion
    54: "Tambourine",
    56: "Cowbell",
    58: "Vibraslap",
    60: "Hi Bongo",
    61: "Low Bongo",
    62: "Mute Hi Conga",
    63: "Open Hi Conga",
    64: "Low Conga",
    65: "High Timbale",
    66: "Low Timbale",
    67: "High Agogo",
    68: "Low Agogo",
    69: "Cabasa",
    70: "Maracas",
    71: "Short Whistle",
    72: "Long Whistle",
    73: "Short Guiro",
    74: "Long Guiro",
    75: "Claves",
    76: "Hi Wood Block",
    77: "Low Wood Block",
    78: "Mute Cuica",
    79: "Open Cuica",
    80: "Mute Triangle",
    81: "Open Triangle",
}

# Common drum kit components for validation
KICK_NOTES = [35, 36]
SNARE_NOTES = [37, 38, 40]
HIHAT_NOTES = [42, 44, 46]
CRASH_NOTES = [49, 52, 55, 57]
RIDE_NOTES = [51, 53, 59]
TOM_NOTES = [41, 43, 45, 47, 48, 50]

# MIDI constants
TICKS_PER_QUARTER_NOTE = 480  # Standard resolution
CHANNEL_DRUMS = 9  # MIDI channel 10 (0-indexed as 9)
```

---

## 6. Export Workflow

Complete workflow from LLM output to MIDI file.

```python
# src/midi/export.py

from pathlib import Path
from typing import Dict, List
import datetime

def export_midi_from_llm(
    llm_output: Dict,
    artist_name: str,
    variation_num: int,
    style_profile: 'StyleProfile',
    output_dir: Path = Path("output/patterns")
) -> Path:
    """
    Complete export workflow: LLM JSON → validated → humanized → MIDI file.
    
    Args:
        llm_output: JSON from LLM generation
        artist_name: For filename
        variation_num: Variation number (1-8)
        style_profile: For style-specific humanization
        output_dir: Where to save files
    
    Returns:
        Path to created MIDI file
    """
    # 1. Validate LLM output
    error = validate_midi_json(llm_output)
    if error:
        raise MIDIValidationError(f"Invalid LLM output: {error}")
    
    # 2. Apply style transfer
    notes = apply_style_transfer(
        llm_output['notes'],
        style_profile.to_style_characteristics()
    )
    llm_output['notes'] = notes
    
    # 3. Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{artist_name}_var{variation_num}_{timestamp}.mid"
    output_path = output_dir / filename
    
    # 4. Convert to MIDI with humanization
    output_path = json_to_midi(
        llm_output,
        output_path,
        humanize=True,
        style_params=style_profile.quantitative_params
    )
    
    # 5. Validate final MIDI file
    validate_midi_file(output_path)
    
    return output_path
```

---

## Quick Reference

### Common Operations

**Generate MIDI from LLM:**
```python
midi_file = json_to_midi(llm_output, Path("output.mid"), humanize=True)
```

**Validate before conversion:**
```python
if error := validate_midi_json(llm_output):
    print(f"Invalid: {error}")
```

**Apply style transfer:**
```python
styled_notes = apply_style_transfer(notes, style_characteristics)
```

**Read MIDI file:**
```python
from mido import MidiFile
mid = MidiFile("pattern.mid")
for track in mid.tracks:
    for msg in track:
        print(msg)
```

---

**MIDI operations remain largely the same, with the main change being input source: LLM JSON instead of PyTorch tokens.**
