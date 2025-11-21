#!/usr/bin/env python3
"""
Flatten BMAD Commands for Claude Code Compatibility

This script converts nested BMAD command structure:
  .claude/commands/bmad/bmm/workflows/document-project.md

To flat structure compatible with Claude Code:
  .claude/commands/bmm-document-project.md

Usage:
    python scripts/flatten_bmad_commands.py [--dry-run]
"""

import argparse
import shutil
from pathlib import Path


def flatten_command_name(nested_path: Path, base_dir: Path) -> str:
    """
    Convert nested path to flat command name.

    Example:
        bmad/bmm/workflows/document-project.md → bmm-document-project.md
        bmad/core/workflows/brainstorming.md → core-brainstorming.md
    """
    # Get relative path from base_dir
    rel_path = nested_path.relative_to(base_dir)

    # Remove 'bmad/' prefix
    parts = list(rel_path.parts)
    if parts[0] == "bmad":
        parts = parts[1:]

    # Join with hyphens, keeping the .md extension
    name = "-".join(parts)

    return name


def main(dry_run: bool = False):
    """Flatten all BMAD commands."""

    project_root = Path(__file__).parent.parent
    commands_dir = project_root / ".claude" / "commands"
    bmad_dir = commands_dir / "bmad"

    if not bmad_dir.exists():
        print(f"❌ BMAD commands directory not found: {bmad_dir}")
        return

    # Find all .md files in bmad subdirectories
    md_files = list(bmad_dir.rglob("*.md"))

    print(f"Found {len(md_files)} BMAD command files")
    print()

    if dry_run:
        print("🔍 DRY RUN - No files will be moved")
        print()

    moved_count = 0
    skipped_count = 0

    for md_file in md_files:
        flat_name = flatten_command_name(md_file, bmad_dir)
        target_path = commands_dir / flat_name

        # Check if target already exists
        if target_path.exists():
            print(f"⚠️  SKIP: {flat_name} (already exists)")
            skipped_count += 1
            continue

        print(f"{'📋' if dry_run else '✅'} {md_file.relative_to(commands_dir)}")
        print(f"   → {flat_name}")

        if not dry_run:
            # Copy file (preserve original for safety)
            shutil.copy2(md_file, target_path)
            moved_count += 1

    print()
    print("=" * 60)

    if dry_run:
        print(f"📊 Would create {len(md_files) - skipped_count} flat commands")
        print(f"   (skipped {skipped_count} existing)")
    else:
        print(f"✅ Created {moved_count} flat commands")
        print(f"⚠️  Skipped {skipped_count} existing")
        print()
        print("Original nested commands preserved in .claude/commands/bmad/")
        print("You can now use commands like:")
        print("  /bmm-document-project")
        print("  /bmm-analyst")
        print("  /core-brainstorming")
        print()
        print("🔄 Restart Claude Code to see the new commands")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten BMAD commands for Claude Code compatibility"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )

    args = parser.parse_args()
    main(dry_run=args.dry_run)
