"""
Script to create test MIDI fixture for unit tests.

Run this once to generate tests/fixtures/test_pattern.mid
"""

from pathlib import Path

import mido


def create_test_drum_pattern():
    """Create a simple drum pattern MIDI file for testing."""
    mid = mido.MidiFile(ticks_per_beat=480)

    # Create drum track
    drum_track = mido.MidiTrack()
    drum_track.name = "Drums"
    mid.tracks.append(drum_track)

    # Add tempo (120 BPM = 500000 microseconds per beat)
    drum_track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    # Add time signature (4/4)
    drum_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    # Build events list with absolute times, then convert to delta
    events = []

    # Create a simple 4-bar pattern
    # Channel 9 = drum channel (0-indexed)
    # Notes: 36=kick, 38=snare, 42=hihat

    for bar in range(4):
        bar_start = bar * 480 * 4  # 4 beats per bar

        for beat in range(4):
            beat_time = bar_start + beat * 480

            # Kick on beats 1 and 3
            if beat in [0, 2]:
                events.append(
                    (beat_time, "note_on", 36, 100 - bar * 2)
                )  # Slight velocity variation

            # Snare on beats 2 and 4
            if beat in [1, 3]:
                events.append((beat_time, "note_on", 38, 110 - bar * 2))

            # Hi-hat on every 8th note
            for eighth in range(2):
                hihat_time = beat_time + eighth * 240
                velocity = 80 if eighth == 0 else 70
                events.append((hihat_time, "note_on", 42, velocity))

    # Sort events by time
    events.sort(key=lambda x: x[0])

    # Convert to delta times and add to track
    last_time = 0
    for abs_time, msg_type, note, velocity in events:
        delta = abs_time - last_time
        drum_track.append(
            mido.Message(msg_type, note=note, velocity=velocity, channel=9, time=delta)
        )
        last_time = abs_time

    # End of track
    drum_track.append(mido.MetaMessage("end_of_track", time=0))

    return mid


if __name__ == "__main__":
    output_path = Path(__file__).parent / "test_pattern.mid"
    mid = create_test_drum_pattern()
    mid.save(str(output_path))
    print(f"Created test MIDI file: {output_path}")
