# 🎵 Drum Pattern Generator - Complete Context Engineering Package

**Modern PyTorch-based MIDI drum pattern generator with optimal Cursor IDE integration**

## 📦 Package Contents

This package contains **complete context engineering documentation** for developing a modern drum pattern generator using Cursor IDE with Claude Code.

### ✅ What's Included

```
MidiDrumiGen/
├── .cursorrules                          # Cursor IDE AI behavior rules (11KB)
├── .cursorcontext/                       # Context engineering documents
│   ├── 01_project_overview.md           # Project goals and features (7KB)
│   ├── 02_architecture.md               # System design (24KB)
│   ├── 03_dependencies.md               # Verified modern libraries (14KB)
│   ├── 04_midi_operations.md            # MIDI domain knowledge (17KB)
│   ├── 05_ml_pipeline.md                # ML training & inference (18KB)
│   └── 06_common_tasks.md               # Task quick reference (13KB)
├── docs/
│   └── cursor_guide.md                  # Complete Cursor usage guide (11KB)
├── scripts/
│   └── verify_installation.py           # Environment verification script
├── requirements.txt                      # All dependencies with versions
└── README.md                             # Project overview

Total Documentation: ~115KB of precise, modern context
```

---

## 🎯 What Problem Does This Solve?

### The Legacy Component Problem

Our previous discussion involved using **Magenta (TensorFlow-based)** models like GrooVAE and MusicVAE. However, the validation research revealed:

❌ **Magenta repository is inactive** (last update July 2025)  
❌ **TensorFlow/PyTorch CUDA conflicts** are unsolvable  
❌ **pretty-midi abandoned** since 2020  
❌ **pylive compatibility** with Ableton Live 12 unverified  
❌ **Python 3.11+ incompatibility** with legacy stack

### The Modern Solution

This package provides:

✅ **PyTorch 2.4+ with CUDA 12.1** (modern, actively maintained)  
✅ **mido 1.3.3** for MIDI (actively maintained)  
✅ **MidiTok 2.1+** for tokenization (framework-agnostic)  
✅ **FastAPI + Celery + Redis** (production-grade stack)  
✅ **Python 3.11** (latest stable)  
✅ **Complete context engineering** for Cursor IDE

---

## 🚀 Quick Start

### 1. Setup Project

**Windows (PowerShell):**
```powershell
# Navigate to project directory
cd C:\Users\<YourUsername>\Documents\Dev\MidiDrumiGen

# Verify structure
Get-ChildItem -Force .cursorrules, .cursorcontext, docs, requirements.txt, README.md
```

**Linux/macOS:**
```bash
# Navigate to project directory
cd ~/Documents/Dev/MidiDrumiGen

# Verify structure
ls -la .cursorrules .cursorcontext/ docs/ requirements.txt README.md
```

### 2. Setup Python Environment

**Windows (PowerShell):**
```powershell
# Requires Python 3.11 (use py launcher)
py -3.11 -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
# If you get an execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install PyTorch (CUDA 12.1)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

**Linux/macOS:**
```bash
# Requires Python 3.11
python3.11 -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA 12.1)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

**Windows/Linux/macOS:**
```bash
python scripts/verify_installation.py
```

**Expected Output:**
```
======================================================================
DRUM PATTERN GENERATOR - INSTALLATION VERIFICATION
======================================================================

Python Version:
  ✓ Python 3.11.x

Core ML Dependencies:
  ✓ torch 2.4.1
  ✓ transformers 4.46.3
  ✓ GPU: NVIDIA RTX 4090 (24.0GB)

MIDI Processing:
  ✓ mido 1.3.3
  ✓ miditoolkit 1.0.1
  ✓ miditok 2.1.8

Backend Infrastructure:
  ✓ fastapi 0.121.0
  ✓ celery 5.5.3
  ✓ redis 7.0.1

Context Engineering Files:
  ✓ .cursorrules (11.0KB)
  ✓ .cursorcontext/01_project_overview.md (7.3KB)
  ✓ .cursorcontext/02_architecture.md (24.8KB)
  ✓ .cursorcontext/03_dependencies.md (13.8KB)
  ✓ .cursorcontext/04_midi_operations.md (17.7KB)
  ✓ .cursorcontext/05_ml_pipeline.md (18.2KB)
  ✓ .cursorcontext/06_common_tasks.md (13.1KB)

======================================================================
✓ ALL CHECKS PASSED (24/24)

Your environment is ready for development!
```

### 4. Open in Cursor IDE

**Windows:**
```powershell
# Open project in Cursor (if cursor command is in PATH)
cursor C:\Users\<YourUsername>\Documents\Dev\MidiDrumiGen

# Or from within Cursor IDE:
# File → Open Folder → Select MidiDrumiGen
```

**Linux/macOS:**
```bash
# Open project in Cursor
cursor ~/Documents/Dev/MidiDrumiGen

# Or from within Cursor:
# File → Open Folder → Select MidiDrumiGen
```

### 5. Development Workflow with Claude Code

**This project is optimized for development with Claude Code in Cursor IDE.**

**Test Context Engineering:**

In Cursor's chat (Claude Code), try:

```
@docs 01_project_overview.md
What is this project about?
```

Expected: Explanation of PyTorch-based drum pattern generator, NO mention of Magenta or TensorFlow

```
@docs 03_dependencies.md
Should I use pretty-midi or mido?
```

Expected: Use mido 1.3.3 (actively maintained), NOT pretty-midi (abandoned 2020)

**Typical Development Workflow:**

1. **Ask Claude Code to implement features:**
   ```
   @docs 02_architecture.md @docs 04_midi_operations.md
   Implement the MIDI export function in src/midi/export.py
   ```

2. **Use context documents for guidance:**
   ```
   @docs 05_ml_pipeline.md
   How should I structure the training loop?
   ```

3. **Let Claude Code write code:**
   - Claude Code has access to all context documents
   - It will follow `.cursorrules` for coding standards
   - It knows to use modern libraries (mido, PyTorch, etc.)

4. **Review and iterate:**
   - Claude Code can explain code
   - It can refactor based on architecture docs
   - It can write tests based on patterns

---

## 📚 Documentation Structure

### Core Context Documents (`.cursorcontext/`)

| Document | Purpose | Use When |
|----------|---------|----------|
| `01_project_overview.md` | High-level goals, features, tech stack | Starting new features, understanding project scope |
| `02_architecture.md` | System design, data flow, components | Designing new features, debugging integration |
| `03_dependencies.md` | Verified libraries, versions, compatibility | Adding dependencies, resolving conflicts |
| `04_midi_operations.md` | MIDI protocol, drum mapping, humanization | Implementing MIDI features, export functions |
| `05_ml_pipeline.md` | Model architecture, training, inference | ML development, training models, optimization |
| `06_common_tasks.md` | CLI commands, workflows, troubleshooting | Daily development tasks, quick reference |

### Configuration Files

| File | Purpose |
|------|---------|
| `.cursorrules` | Defines Cursor IDE AI behavior, coding standards, constraints |
| `requirements.txt` | All dependencies with verified versions |
| `README.md` | Project overview and quick start |

### Guides

| Guide | Purpose |
|-------|---------|
| `docs/cursor_guide.md` | Complete guide for using Cursor IDE with context engineering |
| `scripts/verify_installation.py` | Verification script for dependencies and setup |

---

## 🎓 How to Use with Cursor IDE

### Basic Usage

```
# Load specific context
@docs 02_architecture.md

# Load multiple contexts
@docs 03_dependencies.md @docs 04_midi_operations.md

# Combine with code
@file src/api/main.py @docs 02_architecture.md

# Search project
@folder src/models
```

### Example Workflows

#### Implementing New Feature

```
Step 1: @docs 01_project_overview.md
Does this feature align with project goals?

Step 2: @docs 02_architecture.md
Which component handles this?

Step 3: @docs 03_dependencies.md
What libraries can I use?

Step 4: Implement with context loaded
@folder src/relevant_component @docs 02_architecture.md
```

#### Debugging Issue

```
Step 1: @docs 02_architecture.md
Identify which component is failing

Step 2: @docs 06_common_tasks.md
Check common issues and solutions

Step 3: @file failing_file.py @docs 02_architecture.md
Compare against documented patterns
```

#### Adding MIDI Export

```
@docs 04_midi_operations.md @docs 03_dependencies.md
I need to export model-generated tokens to MIDI with humanization.
Show me the complete implementation using mido.
```

---

## 🔍 Key Features of This Package

### 1. Zero Legacy Components

**Strict Policy:**
- ❌ NO Magenta (TensorFlow-based, inactive)
- ❌ NO pretty-midi (abandoned 2020)
- ❌ NO TensorFlow (conflicts with PyTorch)
- ❌ NO unverified libraries

**Only Modern Alternatives:**
- ✅ PyTorch 2.4+ (actively maintained)
- ✅ mido 1.3.3 (Python 3.11+ support)
- ✅ MidiTok 2.1+ (framework-agnostic)
- ✅ FastAPI 0.121+ (monthly releases)

### 2. Comprehensive Verification

Every dependency is verified for:
- ✅ Python 3.11/3.12 compatibility
- ✅ Active maintenance (last 12 months)
- ✅ CUDA 12.1+ support (where applicable)
- ✅ Production-grade stability

### 3. Context Engineering Best Practices

- **Modular documents** (6 separate files, each ~7-24KB)
- **Clear sectioning** with consistent structure
- **Progressive disclosure** (overview → details)
- **Task-oriented** organization
- **Quick reference** sections
- **Code examples** throughout

### 4. Cursor IDE Optimization

- **`.cursorrules`** defines AI behavior
- **`.cursorcontext/`** provides structured knowledge
- **`@docs` references** for easy loading
- **Multi-document queries** supported
- **File + context** combination patterns

---

## 📋 Verification Checklist

Before starting development:

- [ ] Python 3.11 installed and activated
  - **Windows:** `py -3.11 --version` should show 3.11.x
  - **Linux/macOS:** `python3.11 --version` should show 3.11.x
- [ ] Virtual environment created and activated
  - **Windows:** `.\venv\Scripts\Activate.ps1`
  - **Linux/macOS:** `source venv/bin/activate`
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] PyTorch CUDA available (or CPU mode accepted)
  - Check with: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Redis running (optional for development)
  - **Docker:** `docker run -d -p 6379:6379 --name redis redis:7-alpine`
  - Or install Redis locally
- [ ] All context files present (`.cursorcontext/*.md`)
  - Should have 6 files: 01-06_*.md
- [ ] `.cursorrules` file present
- [ ] Verification script passes (`python scripts/verify_installation.py`)
- [ ] Cursor IDE open with project loaded
- [ ] Claude Code can load context (`@docs 01_project_overview.md` in Cursor chat)

---

## 🎯 What You Can Build Now

With this context engineering setup, you can immediately:

1. **Design System Architecture**
   - Load `02_architecture.md` for patterns
   - Implement FastAPI + Celery + Redis stack
   - Follow documented data flow

2. **Implement MIDI Export**
   - Load `04_midi_operations.md` for GM drum mapping
   - Use mido for MIDI I/O
   - Apply humanization algorithms

3. **Train ML Models**
   - Load `05_ml_pipeline.md` for architecture
   - Use PyTorch 2.4+ with custom Transformer
   - Implement training pipeline with MidiTok

4. **Add New Features**
   - Load `06_common_tasks.md` for workflows
   - Follow established patterns
   - Verify against architecture

5. **Debug Issues**
   - Load `03_dependencies.md` for compatibility
   - Check common issues in `06_common_tasks.md`
   - Compare against documented examples

### 📖 Detailed Development Guide

**For a complete step-by-step guide on starting development, see:**
- **`DEVELOPMENT_START_GUIDE.md`** - Comprehensive walkthrough of implementing all core components
- **`CURSOR_WORKFLOW.md`** - Best practices for using Claude Code in Cursor IDE

The development guide includes:
- ✅ Phase-by-phase implementation plan
- ✅ Specific prompts to use with Claude Code
- ✅ Verification steps for each component
- ✅ Testing procedures
- ✅ Troubleshooting tips

---

## 🚨 Important Reminders

### When Claude Suggests Legacy Code

If Cursor's AI suggests using Magenta, TensorFlow, or pretty-midi:

```
STOP. @docs 03_dependencies.md
We do NOT use legacy libraries. Use modern alternatives:
- PyTorch 2.4+ (not TensorFlow)
- mido 1.3.3 (not pretty-midi)
- Custom models (not Magenta GrooVAE)
```

### Dependency Rules

**Before installing ANY new package:**

```
@docs 03_dependencies.md
Is [package-name] compatible with our stack?
- Python 3.11?
- Actively maintained?
- No conflicts with PyTorch?
```

---

## 📈 Next Steps

### Immediate (Week 1)

1. Setup development environment
2. Familiarize with context documents
3. Implement basic MIDI export
4. Create simple pattern generator

### Short-term (Month 1)

1. Train initial PyTorch model on Groove MIDI Dataset
2. Implement FastAPI endpoints
3. Add Celery task queue
4. Create basic humanization

### Medium-term (Quarter 1)

1. Add multiple producer styles
2. Implement latent space manipulation
3. Build Ableton Live integration
4. Create web UI

---

## 💡 Pro Tips

1. **Always start with context:**
   Load relevant `.md` files before coding

2. **Progressive context loading:**
   Start broad (overview), then narrow (specific docs)

3. **Verify against docs:**
   Cross-check implementations with documented patterns

4. **Update docs as you go:**
   These are living documents - improve them

5. **Use verification script:**
   Run `verify_installation.py` after changes

---

## 📞 Support & Resources

### Documentation

- **Complete guide:** `docs/cursor_guide.md`
- **Context docs:** `.cursorcontext/*.md` (6 files)
- **Project README:** `README.md`

### Verification

```bash
# Check installation
python scripts/verify_installation.py

# Test Cursor context
# In Cursor: @docs 01_project_overview.md
```

---

## 🎊 Success Criteria

You know the setup is working when:

✅ Verification script passes all checks  
✅ Cursor can load `@docs` references  
✅ Claude knows to use PyTorch (not TensorFlow)  
✅ Claude recommends mido (not pretty-midi)  
✅ Claude references modern Python 3.11 features  
✅ Context documents answer your questions  
✅ Code follows documented architecture patterns

---

## 📝 Summary

**This package provides:**

✅ **Modern tech stack** (PyTorch, mido, FastAPI)  
✅ **Zero legacy components** (no Magenta, TensorFlow)  
✅ **Verified dependencies** (Python 3.11, CUDA 12.1)  
✅ **Complete context engineering** (115KB documentation)  
✅ **Cursor IDE optimization** (`.cursorrules`, `.cursorcontext/`)  
✅ **Production-ready patterns** (architecture, workflows)  
✅ **Domain knowledge** (MIDI, ML, drums)  
✅ **Quick reference** (commands, troubleshooting)

**Start developing immediately with confidence that:**
- All libraries are modern and maintained
- All patterns are production-grade
- All documentation is up-to-date
- Cursor AI has optimal context
- No legacy components will be suggested

---

**Ready to build? Open in Cursor and start with:**

```
@docs 01_project_overview.md
Let's build this drum pattern generator!
```

🎵 Happy coding! 🎵
