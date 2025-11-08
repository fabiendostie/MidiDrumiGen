"""MIDI constants and drum mappings."""

# Core drums used in most patterns
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

# GM Drum Kit (Channel 10)
GM_DRUM_MAP = {
    35: 'Acoustic Bass Drum',
    36: 'Bass Drum 1',
    37: 'Side Stick',
    38: 'Acoustic Snare',
    39: 'Hand Clap',
    40: 'Electric Snare',
    41: 'Low Floor Tom',
    42: 'Closed Hi-Hat',
    43: 'High Floor Tom',
    44: 'Pedal Hi-Hat',
    45: 'Low Tom',
    46: 'Open Hi-Hat',
    47: 'Low-Mid Tom',
    48: 'Hi-Mid Tom',
    49: 'Crash Cymbal 1',
    50: 'High Tom',
    51: 'Ride Cymbal 1',
    52: 'Chinese Cymbal',
    53: 'Ride Bell',
    54: 'Tambourine',
    55: 'Splash Cymbal',
    56: 'Cowbell',
    57: 'Crash Cymbal 2',
    58: 'Vibraslap',
    59: 'Ride Cymbal 2',
    60: 'Hi Bongo',
    61: 'Low Bongo',
    62: 'Mute Hi Conga',
    63: 'Open Hi Conga',
    64: 'Low Conga',
    65: 'High Timbale',
    66: 'Low Timbale',
    67: 'High Agogo',
    68: 'Low Agogo',
    69: 'Cabasa',
    70: 'Maracas',
    71: 'Short Whistle',
    72: 'Long Whistle',
    73: 'Short Guiro',
    74: 'Long Guiro',
    75: 'Claves',
    76: 'Hi Wood Block',
    77: 'Low Wood Block',
    78: 'Mute Cuica',
    79: 'Open Cuica',
    80: 'Mute Triangle',
    81: 'Open Triangle',
}

# Valid drum note range
MIN_DRUM_NOTE = 35
MAX_DRUM_NOTE = 81

# MIDI constants
DRUMS_CHANNEL = 9  # Channel 10 (0-indexed)
DEFAULT_TICKS_PER_BEAT = 480

