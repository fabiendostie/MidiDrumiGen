#!/usr/bin/env python3
"""
Installation Verification Script
Checks all dependencies and context documents are properly set up.
"""

import sys
import importlib
import io
from pathlib import Path
from typing import Tuple

# Fix Windows console encoding to support Unicode checkmarks
if sys.platform == 'win32':
    # Try to reconfigure stdout with UTF-8 encoding
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        # If reconfigure fails, wrap stdout with UTF-8 TextIOWrapper
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except Exception:
            pass  # Fall back to default encoding

# Unicode symbols with ASCII fallback
try:
    # Test if we can print Unicode
    test = "✓ ✗ ⚠".encode(sys.stdout.encoding or 'utf-8')
    CHECK = "✓"
    CROSS = "✗"
    WARN = "⚠"
except (UnicodeEncodeError, LookupError, AttributeError):
    # Fallback to ASCII
    CHECK = "[OK]"
    CROSS = "[FAIL]"
    WARN = "[WARN]"

def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.11"""
    version = sys.version_info
    if version.major == 3 and version.minor == 11:
        return True, f"{CHECK} Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"{CROSS} Python {version.major}.{version.minor}.{version.micro} (need 3.11)"

def check_package(name: str, version_check: str = None) -> Tuple[bool, str]:
    """Check if package is installed with optional version check"""
    try:
        module = importlib.import_module(name)
        if hasattr(module, '__version__'):
            ver = module.__version__
            if version_check and not ver.startswith(version_check):
                return False, f"{CROSS} {name} {ver} (expected {version_check}+)"
            return True, f"{CHECK} {name} {ver}"
        else:
            return True, f"{CHECK} {name} (version unknown)"
    except ImportError:
        return False, f"{CROSS} {name} not installed"

def check_file(filepath: str) -> Tuple[bool, str]:
    """Check if file exists"""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size / 1024  # KB
        return True, f"{CHECK} {filepath} ({size:.1f}KB)"
    else:
        return False, f"{CROSS} {filepath} missing"

def main():
    print("=" * 70)
    print("DRUM PATTERN GENERATOR - INSTALLATION VERIFICATION")
    print("=" * 70)
    print()
    
    # Track results
    all_checks = []
    
    # Python version
    print("Python Version:")
    result = check_python_version()
    all_checks.append(result)
    print(f"  {result[1]}")
    print()
    
    # Core ML dependencies
    print("Core ML Dependencies:")
    ml_packages = [
        ('torch', '2.4'),
        ('transformers', '4.46'),
    ]
    for pkg, ver in ml_packages:
        result = check_package(pkg, ver)
        all_checks.append(result)
        print(f"  {result[1]}")
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  {CHECK} GPU: {device_name} ({memory_gb:.1f}GB)")
        else:
            print(f"  {WARN} GPU: Not available (CPU mode)")
    except:
        print(f"  {CROSS} GPU: Cannot check")
    print()
    
    # MIDI libraries
    print("MIDI Processing:")
    midi_packages = [
        ('mido', '1.3'),
        ('miditoolkit', '1.0'),
        ('miditok', '2.1'),
    ]
    for pkg, ver in midi_packages:
        result = check_package(pkg, ver)
        all_checks.append(result)
        print(f"  {result[1]}")
    print()
    
    # Backend infrastructure
    print("Backend Infrastructure:")
    backend_packages = [
        ('fastapi', '0.121'),
        ('celery', '5.5'),
        ('redis', '7.0'),
    ]
    for pkg, ver in backend_packages:
        result = check_package(pkg, ver)
        all_checks.append(result)
        print(f"  {result[1]}")
    print()
    
    # Check Redis connection
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, socket_timeout=1)
        r.ping()
        print(f"  {CHECK} Redis connection OK")
    except:
        print(f"  {WARN} Redis connection failed (start: docker run -d -p 6379:6379 redis:7-alpine)")
    print()
    
    # Context engineering files
    print("Context Engineering Files:")
    context_files = [
        '.cursorrules',
        '.cursorcontext/01_project_overview.md',
        '.cursorcontext/02_architecture.md',
        '.cursorcontext/03_dependencies.md',
        '.cursorcontext/04_midi_operations.md',
        '.cursorcontext/05_ml_pipeline.md',
        '.cursorcontext/06_common_tasks.md',
    ]
    for filepath in context_files:
        result = check_file(filepath)
        all_checks.append(result)
        print(f"  {result[1]}")
    print()
    
    # Documentation
    print("Documentation:")
    doc_files = [
        'README.md',
        'docs/cursor_guide.md',
        'requirements.txt',
    ]
    for filepath in doc_files:
        result = check_file(filepath)
        all_checks.append(result)
        print(f"  {result[1]}")
    print()
    
    # Summary
    print("=" * 70)
    passed = sum(1 for ok, _ in all_checks if ok)
    total = len(all_checks)

    if passed == total:
        print(f"{CHECK} ALL CHECKS PASSED ({passed}/{total})")
        print()
        print("Your environment is ready for development!")
        print()
        print("Next steps:")
        print("1. Open project in Cursor IDE")
        print("2. Review docs/cursor_guide.md")
        print("3. Start development with context engineering")
        return 0
    else:
        print(f"{WARN} {total - passed} CHECKS FAILED ({passed}/{total})")
        print()
        print("Please fix the issues above and run again.")
        print()
        print("Quick fixes:")
        print("- Python 3.11: py -3.11 -m venv venv && venv\\Scripts\\activate")
        print("- Dependencies: pip install -r requirements.txt")
        print("- Redis: docker run -d -p 6379:6379 redis:7-alpine")
        return 1

if __name__ == '__main__':
    sys.exit(main())
