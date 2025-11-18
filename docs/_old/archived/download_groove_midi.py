"""Download and extract Groove MIDI Dataset."""

import urllib.request
import zipfile
import logging
from pathlib import Path
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Groove MIDI Dataset URL
GROOVE_MIDI_URL = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
OUTPUT_DIR = Path("data/groove_midi")
TEMP_ZIP = Path("data/groove_midi.zip")


def download_file(url: str, output_path: Path) -> bool:
    """
    Download file with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Downloading from {url}")
        logger.info(f"Saving to {output_path}")
        
        def progress_hook(count, block_size, total_size):
            """Show download progress."""
            percent = int(count * block_size * 100 / total_size)
            if count % 50 == 0:  # Update every 50 blocks
                logger.info(f"Progress: {percent}%")
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        logger.info("Download complete!")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """
    Extract ZIP file, skipping problematic files.
    
    Args:
        zip_path: Path to ZIP file
        output_dir: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Extracting {zip_path} to {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_count = 0
        skipped_count = 0
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Skip macOS resource fork files
                if 'Icon' in member or '__MACOSX' in member or member.endswith('.DS_Store'):
                    skipped_count += 1
                    continue
                
                try:
                    zip_ref.extract(member, output_dir)
                    extracted_count += 1
                except Exception as e:
                    logger.warning(f"Skipped problematic file: {member} ({e})")
                    skipped_count += 1
        
        logger.info(f"Extraction complete! Extracted {extracted_count} files, skipped {skipped_count}")
        return True
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def count_midi_files(directory: Path) -> int:
    """Count MIDI files in directory."""
    midi_files = list(directory.glob("**/*.mid")) + list(directory.glob("**/*.midi"))
    return len(midi_files)


def main():
    """Main download and extraction script."""
    logger.info("=" * 80)
    logger.info("Groove MIDI Dataset Downloader")
    logger.info("=" * 80)
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Check if already downloaded
    if OUTPUT_DIR.exists():
        midi_count = count_midi_files(OUTPUT_DIR)
        if midi_count > 0:
            logger.info(f"Groove MIDI dataset already exists in {OUTPUT_DIR}")
            logger.info(f"Found {midi_count} MIDI files")
            
            response = input("Do you want to re-download? (y/N): ")
            if response.lower() != 'y':
                logger.info("Using existing dataset")
                sys.exit(0)
    
    # Download
    logger.info(f"Downloading Groove MIDI Dataset...")
    logger.info(f"This may take several minutes (dataset is ~100MB)")
    
    success = download_file(GROOVE_MIDI_URL, TEMP_ZIP)
    if not success:
        logger.error("Download failed!")
        sys.exit(1)
    
    # Extract
    logger.info(f"Extracting dataset...")
    success = extract_zip(TEMP_ZIP, OUTPUT_DIR)
    if not success:
        logger.error("Extraction failed!")
        sys.exit(1)
    
    # Clean up ZIP file
    logger.info("Cleaning up temporary files...")
    TEMP_ZIP.unlink()
    
    # Verify
    midi_count = count_midi_files(OUTPUT_DIR)
    logger.info(f"Verification: Found {midi_count} MIDI files in {OUTPUT_DIR}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"Dataset location: {OUTPUT_DIR}")
    print(f"MIDI files found: {midi_count}")
    print(f"Next step: Run tokenization script")
    print("=" * 80)
    
    if midi_count == 0:
        logger.error("No MIDI files found after extraction!")
        sys.exit(1)


if __name__ == "__main__":
    main()

