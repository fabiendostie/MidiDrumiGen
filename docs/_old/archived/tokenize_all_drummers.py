"""Batch tokenize all Groove MIDI drummers with style labels."""

import logging
from pathlib import Path
import json
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_style_labels(labels_path: Path) -> dict:
    """Load style mapping from JSON file."""
    with open(labels_path) as f:
        data = json.load(f)
    return data['drummers'], data['style_mapping']


def tokenize_drummer(
    drummer_name: str,
    style_name: str,
    input_dir: Path,
    output_dir: Path,
    tokenizer_path: Path,
) -> bool:
    """
    Tokenize all MIDI files for a single drummer.
    
    Args:
        drummer_name: Name of drummer directory (e.g., "drummer1")
        style_name: Producer style to assign
        input_dir: Path to drummer's directory
        output_dir: Output directory for tokens
        tokenizer_path: Path to save tokenizer (only first drummer)
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Tokenizing {drummer_name} -> style '{style_name}'")
    
    # Build command
    cmd = [
        sys.executable,  # Python interpreter
        "scripts/tokenize_dataset.py",
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--style", style_name,
        "--vocab-size", "1000",
        "--beat-resolution", "8",
        "--num-velocities", "32",
    ]
    
    # Save tokenizer on first drummer only
    if drummer_name == "drummer1":
        cmd.extend(["--save-tokenizer", str(tokenizer_path)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"✓ {drummer_name} tokenized successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {drummer_name} tokenization failed:")
        logger.error(f"  {e.stderr}")
        return False


def main():
    """Main batch tokenization script."""
    logger.info("=" * 80)
    logger.info("Batch Tokenization - All Groove MIDI Drummers")
    logger.info("=" * 80)
    
    # Paths
    groove_dir = Path("data/groove_midi/groove")
    output_dir = Path("data/tokenized")
    tokenizer_path = Path("models/tokenizers/remi_tokenizer")
    labels_path = Path("data/style_labels.json")
    
    # Validate inputs
    if not groove_dir.exists():
        logger.error(f"Groove MIDI directory not found: {groove_dir}")
        logger.error("Please download the dataset first using scripts/download_groove_midi.py")
        sys.exit(1)
    
    if not labels_path.exists():
        logger.error(f"Style labels file not found: {labels_path}")
        sys.exit(1)
    
    # Load style mapping
    drummer_styles, style_mapping = load_style_labels(labels_path)
    logger.info(f"Loaded style mapping for {len(drummer_styles)} drummers")
    logger.info(f"Styles: {list(style_mapping.keys())}")
    
    # Find all drummer directories
    drummer_dirs = sorted([d for d in groove_dir.iterdir() if d.is_dir() and d.name.startswith("drummer")])
    
    if len(drummer_dirs) == 0:
        logger.error("No drummer directories found!")
        sys.exit(1)
    
    logger.info(f"Found {len(drummer_dirs)} drummer directories")
    logger.info("")
    
    # Tokenize each drummer
    results = {}
    
    for drummer_dir in drummer_dirs:
        drummer_name = drummer_dir.name
        
        # Get style for this drummer
        if drummer_name not in drummer_styles:
            logger.warning(f"No style mapping for {drummer_name}, skipping")
            continue
        
        style_name = drummer_styles[drummer_name]['style_name']
        
        # Tokenize
        success = tokenize_drummer(
            drummer_name=drummer_name,
            style_name=style_name,
            input_dir=drummer_dir,
            output_dir=output_dir,
            tokenizer_path=tokenizer_path,
        )
        
        results[drummer_name] = {
            'success': success,
            'style': style_name,
        }
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH TOKENIZATION SUMMARY")
    print("=" * 80)
    
    successful = sum(1 for r in results.values() if r['success'])
    failed = sum(1 for r in results.values() if not r['success'])
    
    print(f"Drummers processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print("")
    
    # Show style distribution
    style_counts = {}
    for info in results.values():
        if info['success']:
            style = info['style']
            style_counts[style] = style_counts.get(style, 0) + 1
    
    print("Style distribution:")
    for style, count in sorted(style_counts.items()):
        print(f"  {style}: {count} drummers")
    
    print("")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer saved: {tokenizer_path}")
    
    # Count tokenized files
    tokenized_files = list(output_dir.glob("*.pt"))
    print(f"Total tokenized files: {len(tokenized_files)}")
    print("=" * 80)
    
    if successful == 0:
        logger.error("No drummers were successfully tokenized!")
        sys.exit(1)
    
    logger.info("✓ Batch tokenization complete!")


if __name__ == "__main__":
    main()

