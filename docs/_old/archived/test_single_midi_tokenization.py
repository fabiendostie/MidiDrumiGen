"""Test tokenizing a single MIDI file to debug the issue."""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from miditok import REMI, TokenizerConfig
import mido

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Pick a single MIDI file
midi_file = Path("data/groove_midi/groove/drummer1/eval_session/10_soul-groove10_102_beat_4-4.mid")

if not midi_file.exists():
    logger.error(f"Test MIDI file not found: {midi_file}")
    sys.exit(1)

logger.info(f"Testing tokenization with: {midi_file}")

# Load with mido first to inspect
mid = mido.MidiFile(midi_file)
logger.info(f"Loaded MIDI file:")
logger.info(f"  Type: {mid.type}")
logger.info(f"  Ticks per beat: {mid.ticks_per_beat}")
logger.info(f"  Number of tracks: {len(mid.tracks)}")

for i, track in enumerate(mid.tracks):
    logger.info(f"  Track {i}: {len(track)} messages")
    note_ons = sum(1 for msg in track if msg.type == 'note_on' and msg.velocity > 0)
    logger.info(f"    Note on events: {note_ons}")

# Try tokenizing with REMI
logger.info("\nTrying REMI tokenization...")

config = TokenizerConfig(
    use_chords=False,
    use_programs=False,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    beat_res={(0, 4): 8},  # 16th note resolution
    num_velocities=32,
    special_tokens=["PAD", "BOS", "EOS"],
)

tokenizer = REMI(config)
logger.info(f"Tokenizer vocab size: {len(tokenizer)}")

# Tokenize
try:
    tokens = tokenizer(str(midi_file))
    logger.info(f"\nTokenization result:")
    logger.info(f"  Type: {type(tokens)}")
    if isinstance(tokens, list):
        logger.info(f"  Length: {len(tokens)}")
        if len(tokens) > 0:
            logger.info(f"  First element type: {type(tokens[0])}")
            logger.info(f"  First element: {tokens[0]}")
    else:
        logger.info(f"  Tokens: {tokens}")
        
    # Convert to tensor if needed
    import torch
    if isinstance(tokens, list) and len(tokens) > 0:
        if isinstance(tokens[0], list):
            tokens_tensor = torch.tensor(tokens[0], dtype=torch.long)
        else:
            tokens_tensor = torch.tensor(tokens, dtype=torch.long)
    else:
        tokens_tensor = tokens
        
    logger.info(f"\nFinal tensor:")
    logger.info(f"  Shape: {tokens_tensor.shape}")
    logger.info(f"  Length: {len(tokens_tensor)}")
    logger.info(f"  First 20 tokens: {tokens_tensor[:20].tolist()}")
    
except Exception as e:
    logger.error(f"Tokenization failed: {e}", exc_info=True)

