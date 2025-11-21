"""Unit tests for MIDI I/O module."""

from pathlib import Path

import mido
import pytest

from src.midi.constants import DEFAULT_TICKS_PER_BEAT, DRUMS_CHANNEL
from src.midi.io import (
    add_note,
    beats_to_ticks,
    create_midi_file,
    read_midi_file,
    save_midi_file,
)


class TestCreateMidiFile:
    """Tests for create_midi_file function."""

    def test_create_midi_file_basic(self):
        """Test basic MIDI file creation."""
        mid, track = create_midi_file()

        assert isinstance(mid, mido.MidiFile)
        assert isinstance(track, mido.MidiTrack)
        assert mid.ticks_per_beat == DEFAULT_TICKS_PER_BEAT

    def test_create_midi_file_custom_tempo(self):
        """Test MIDI file creation with custom tempo."""
        mid, track = create_midi_file(tempo=95)

        # Check for tempo meta message
        tempo_msgs = [msg for msg in track if msg.type == "set_tempo"]
        assert len(tempo_msgs) > 0

        # Verify tempo value
        expected_tempo = mido.bpm2tempo(95)
        assert tempo_msgs[0].tempo == expected_tempo

    def test_create_midi_file_custom_time_signature(self):
        """Test MIDI file creation with custom time signature."""
        mid, track = create_midi_file(time_signature=(3, 4))

        # Check for time signature meta message
        ts_msgs = [msg for msg in track if msg.type == "time_signature"]
        assert len(ts_msgs) > 0
        assert ts_msgs[0].numerator == 3
        assert ts_msgs[0].denominator == 4

    def test_create_midi_file_custom_ticks_per_beat(self):
        """Test MIDI file creation with custom resolution."""
        mid, track = create_midi_file(ticks_per_beat=960)

        assert mid.ticks_per_beat == 960

    def test_create_midi_file_custom_track_name(self):
        """Test MIDI file creation with custom track name."""
        mid, track = create_midi_file(track_name="Test Track")

        # Check for track name meta message
        name_msgs = [msg for msg in track if msg.type == "track_name"]
        assert len(name_msgs) > 0
        assert name_msgs[0].name == "Test Track"

    def test_create_midi_file_sets_drum_channel(self):
        """Test that drum channel (10) is set."""
        mid, track = create_midi_file()

        # Check for program change on channel 9 (drums)
        pc_msgs = [msg for msg in track if msg.type == "program_change"]
        assert len(pc_msgs) > 0
        assert pc_msgs[0].channel == DRUMS_CHANNEL

    def test_create_midi_file_track_added_to_file(self):
        """Test that track is added to MIDI file."""
        mid, track = create_midi_file()

        assert len(mid.tracks) > 0
        assert track in mid.tracks

    def test_create_midi_file_has_all_metadata(self):
        """Test that all required metadata is present."""
        mid, track = create_midi_file(tempo=120, time_signature=(4, 4), track_name="Drums")

        # Should have track name, tempo, time signature, program change
        message_types = [msg.type for msg in track]

        assert "track_name" in message_types
        assert "set_tempo" in message_types
        assert "time_signature" in message_types
        assert "program_change" in message_types


class TestAddNote:
    """Tests for add_note function."""

    def test_add_note_basic(self):
        """Test adding a basic note."""
        mid, track = create_midi_file()
        initial_length = len(track)

        add_note(track, note=36, velocity=100, start_time=0, duration=480)

        # Should have added 2 messages (note on + note off)
        assert len(track) > initial_length

    def test_add_note_creates_note_on_and_off(self):
        """Test that both note on and note off messages are created."""
        mid, track = create_midi_file()

        add_note(track, note=36, velocity=100, start_time=0, duration=480)

        # Get last two messages (should be note on/off)
        note_on = None
        note_off = None
        for msg in reversed(track):
            if msg.type == "note_on" and note_on is None:
                note_on = msg
            elif msg.type == "note_off" and note_off is None:
                note_off = msg

        assert note_on is not None
        assert note_off is not None

    def test_add_note_with_custom_velocity(self):
        """Test adding note with custom velocity."""
        mid, track = create_midi_file()

        add_note(track, note=36, velocity=90, start_time=0, duration=480)

        # Find the note_on message
        note_on_msgs = [msg for msg in track if msg.type == "note_on" and msg.note == 36]
        assert len(note_on_msgs) > 0
        assert note_on_msgs[-1].velocity == 90

    def test_add_note_with_custom_pitch(self):
        """Test adding note with different pitches."""
        mid, track = create_midi_file()

        add_note(track, note=38, velocity=100, start_time=0, duration=480)

        note_on_msgs = [msg for msg in track if msg.type == "note_on" and msg.note == 38]
        assert len(note_on_msgs) > 0

    def test_add_note_with_custom_channel(self):
        """Test adding note with custom channel."""
        mid, track = create_midi_file()

        add_note(track, note=36, velocity=100, start_time=0, duration=480, channel=5)

        # Find note messages with channel 5
        channel_5_msgs = [msg for msg in track if hasattr(msg, "channel") and msg.channel == 5]
        assert len(channel_5_msgs) > 0

    def test_add_note_multiple_notes(self):
        """Test adding multiple notes."""
        mid, track = create_midi_file()

        add_note(track, note=36, velocity=100, start_time=0, duration=480)
        add_note(track, note=38, velocity=90, start_time=480, duration=480)
        add_note(track, note=42, velocity=80, start_time=960, duration=480)

        # Should have multiple note messages
        note_msgs = [msg for msg in track if msg.type in ["note_on", "note_off"]]
        assert len(note_msgs) >= 6  # 3 notes * 2 messages each


class TestReadMidiFile:
    """Tests for read_midi_file function."""

    def test_read_midi_file_basic(self, temp_dir):
        """Test reading a MIDI file."""
        # Create a test MIDI file
        mid, track = create_midi_file()
        add_note(track, note=36, velocity=100, start_time=0, duration=480)
        track.append(mido.MetaMessage("end_of_track", time=0))

        file_path = temp_dir / "test.mid"
        mid.save(file_path)

        # Read it back
        loaded_mid = read_midi_file(file_path)

        assert isinstance(loaded_mid, mido.MidiFile)
        assert loaded_mid.ticks_per_beat == DEFAULT_TICKS_PER_BEAT

    def test_read_nonexistent_file_raises_error(self, temp_dir):
        """Test that reading nonexistent file raises error."""
        file_path = temp_dir / "nonexistent.mid"

        with pytest.raises((FileNotFoundError, OSError, IOError)):
            read_midi_file(file_path)


class TestSaveMidiFile:
    """Tests for save_midi_file function."""

    def test_save_midi_file_basic(self, temp_dir):
        """Test saving a MIDI file."""
        mid, track = create_midi_file()
        add_note(track, note=36, velocity=100, start_time=0, duration=480)
        track.append(mido.MetaMessage("end_of_track", time=0))

        output_path = temp_dir / "output.mid"
        result = save_midi_file(mid, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_save_midi_file_creates_valid_file(self, temp_dir):
        """Test that saved file can be read back."""
        mid, track = create_midi_file(tempo=95)
        add_note(track, note=36, velocity=100, start_time=0, duration=480)
        track.append(mido.MetaMessage("end_of_track", time=0))

        output_path = temp_dir / "valid.mid"
        save_midi_file(mid, output_path)

        # Read it back
        loaded_mid = mido.MidiFile(output_path)
        assert loaded_mid.ticks_per_beat == DEFAULT_TICKS_PER_BEAT

    def test_save_midi_file_with_path_object(self, temp_dir):
        """Test saving with Path object."""
        mid, track = create_midi_file()
        add_note(track, note=36, velocity=100, start_time=0, duration=480)
        track.append(mido.MetaMessage("end_of_track", time=0))

        output_path = Path(temp_dir) / "path_obj.mid"
        save_midi_file(mid, output_path)

        assert output_path.exists()


class TestBeatsToTicks:
    """Tests for beats_to_ticks function."""

    def test_beats_to_ticks_basic(self):
        """Test basic conversion."""
        ticks = beats_to_ticks(1, DEFAULT_TICKS_PER_BEAT)
        assert ticks == DEFAULT_TICKS_PER_BEAT

    def test_beats_to_ticks_multiple_beats(self):
        """Test conversion with multiple beats."""
        ticks = beats_to_ticks(4, DEFAULT_TICKS_PER_BEAT)
        assert ticks == 4 * DEFAULT_TICKS_PER_BEAT

    def test_beats_to_ticks_fractional_beats(self):
        """Test conversion with fractional beats."""
        ticks = beats_to_ticks(0.5, DEFAULT_TICKS_PER_BEAT)
        assert ticks == DEFAULT_TICKS_PER_BEAT // 2

    def test_beats_to_ticks_custom_ticks_per_beat(self):
        """Test conversion with custom resolution."""
        ticks = beats_to_ticks(2, ticks_per_beat=960)
        assert ticks == 1920

    def test_beats_to_ticks_zero_beats(self):
        """Test conversion with zero beats."""
        ticks = beats_to_ticks(0, DEFAULT_TICKS_PER_BEAT)
        assert ticks == 0

    def test_beats_to_ticks_returns_int(self):
        """Test that function returns integer."""
        ticks = beats_to_ticks(1.5, DEFAULT_TICKS_PER_BEAT)
        assert isinstance(ticks, int)


class TestMidiIOIntegration:
    """Integration tests for MIDI I/O operations."""

    def test_create_save_read_workflow(self, temp_dir):
        """Test complete workflow: create, save, and read."""
        # Create MIDI file
        mid, track = create_midi_file(tempo=120)

        # Add notes
        add_note(track, note=36, velocity=100, start_time=0, duration=480)
        add_note(track, note=38, velocity=90, start_time=480, duration=480)
        add_note(track, note=42, velocity=80, start_time=960, duration=480)

        # Add end of track
        track.append(mido.MetaMessage("end_of_track", time=0))

        # Save file
        file_path = temp_dir / "workflow.mid"
        save_midi_file(mid, file_path)

        # Read file back
        loaded_mid = read_midi_file(file_path)

        # Verify properties
        assert loaded_mid.ticks_per_beat == DEFAULT_TICKS_PER_BEAT
        assert len(loaded_mid.tracks) > 0

    def test_create_multiple_tracks(self):
        """Test creating MIDI with multiple tracks."""
        mid = mido.MidiFile(ticks_per_beat=DEFAULT_TICKS_PER_BEAT)

        track1 = mido.MidiTrack()
        track2 = mido.MidiTrack()

        mid.tracks.append(track1)
        mid.tracks.append(track2)

        assert len(mid.tracks) == 2

    def test_create_with_all_drum_types(self, temp_dir):
        """Test creating MIDI with various drum types."""
        mid, track = create_midi_file()

        # Add different drum types
        drum_notes = [36, 38, 42, 46, 49, 51]  # Kick, snare, hi-hats, crash, ride
        for i, note in enumerate(drum_notes):
            add_note(track, note=note, velocity=100, start_time=i * 480, duration=240)

        track.append(mido.MetaMessage("end_of_track", time=0))

        # Save and verify
        file_path = temp_dir / "all_drums.mid"
        save_midi_file(mid, file_path)

        loaded_mid = read_midi_file(file_path)
        note_on_msgs = [msg for msg in loaded_mid.tracks[0] if msg.type == "note_on"]

        # Should have all drum types
        notes_in_file = {msg.note for msg in note_on_msgs}
        assert set(drum_notes).issubset(notes_in_file)

    def test_save_to_nested_directory(self, temp_dir):
        """Test saving to nested directory."""
        mid, track = create_midi_file()
        add_note(track, note=36, velocity=100, start_time=0, duration=480)
        track.append(mido.MetaMessage("end_of_track", time=0))

        nested_path = temp_dir / "subdir" / "nested" / "test.mid"
        nested_path.parent.mkdir(parents=True, exist_ok=True)

        save_midi_file(mid, nested_path)

        assert nested_path.exists()


class TestMidiIOEdgeCases:
    """Tests for edge cases in MIDI I/O."""

    def test_create_with_very_fast_tempo(self):
        """Test creating MIDI with very fast tempo."""
        mid, track = create_midi_file(tempo=240)

        tempo_msgs = [msg for msg in track if msg.type == "set_tempo"]
        assert len(tempo_msgs) > 0

    def test_create_with_very_slow_tempo(self):
        """Test creating MIDI with very slow tempo."""
        mid, track = create_midi_file(tempo=40)

        tempo_msgs = [msg for msg in track if msg.type == "set_tempo"]
        assert len(tempo_msgs) > 0

    def test_create_with_unusual_time_signature(self):
        """Test creating MIDI with unusual time signature."""
        mid, track = create_midi_file(time_signature=(7, 8))

        ts_msgs = [msg for msg in track if msg.type == "time_signature"]
        assert len(ts_msgs) > 0
        assert ts_msgs[0].numerator == 7
        assert ts_msgs[0].denominator == 8

    def test_add_note_with_zero_duration(self):
        """Test adding note with very short duration."""
        mid, track = create_midi_file()

        add_note(track, note=36, velocity=100, start_time=0, duration=1)

        note_msgs = [msg for msg in track if msg.type in ["note_on", "note_off"]]
        assert len(note_msgs) >= 2

    def test_add_note_with_max_velocity(self):
        """Test adding note with maximum velocity."""
        mid, track = create_midi_file()

        add_note(track, note=36, velocity=127, start_time=0, duration=480)

        note_on_msgs = [msg for msg in track if msg.type == "note_on" and msg.note == 36]
        assert note_on_msgs[-1].velocity == 127

    def test_beats_to_ticks_large_number(self):
        """Test conversion with very large number of beats."""
        ticks = beats_to_ticks(1000, DEFAULT_TICKS_PER_BEAT)
        assert ticks == 1000 * DEFAULT_TICKS_PER_BEAT
