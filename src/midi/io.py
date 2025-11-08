"""MIDI file I/O operations using mido."""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from pathlib import Path
from typing import Tuple, List, Dict
from .constants import DRUMS_CHANNEL, DEFAULT_TICKS_PER_BEAT


def create_midi_file(
    tempo: int = 120,
    time_signature: Tuple[int, int] = (4, 4),
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    track_name: str = "Drums"
) -> Tuple[MidiFile, MidiTrack]:
    """
    Create a new MIDI file with proper setup.
    
    Args:
        tempo: BPM
        time_signature: (numerator, denominator)
        ticks_per_beat: MIDI resolution
        track_name: Name for the track
    
    Returns:
        (MidiFile, MidiTrack) tuple
    """
    # Initialize file
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Add track name
    track.append(MetaMessage('track_name', name=track_name, time=0))
    
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
    track.append(Message('program_change', program=0, channel=DRUMS_CHANNEL, time=0))
    
    return mid, track


def add_note(
    track: MidiTrack,
    note: int,
    velocity: int,
    start_time: int,
    duration: int,
    channel: int = DRUMS_CHANNEL
) -> None:
    """
    Add a note to the track.
    
    Args:
        track: MIDI track to add note to
        note: MIDI note number
        velocity: Note velocity (1-127)
        start_time: Start time in ticks
        duration: Note duration in ticks
        channel: MIDI channel (default: 9 for drums)
    """
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


def read_midi_file(midi_path: Path) -> MidiFile:
    """
    Read a MIDI file.
    
    Args:
        midi_path: Path to MIDI file
    
    Returns:
        MidiFile object
    """
    return mido.MidiFile(midi_path)


def save_midi_file(mid: MidiFile, output_path: Path) -> Path:
    """
    Save MIDI file to disk.
    
    Args:
        mid: MidiFile object
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    mid.save(output_path)
    return output_path


def beats_to_ticks(beats: float, ticks_per_beat: int) -> int:
    """
    Convert beats to MIDI ticks.
    
    Args:
        beats: Number of beats
        ticks_per_beat: Ticks per quarter note
    
    Returns:
        Number of ticks
    """
    return int(beats * ticks_per_beat)

