# Cursor IDE & Claude Code Development Workflow

This document describes the recommended workflow for developing this project using Cursor IDE with Claude Code.

## Overview

**This project is designed to be developed primarily with Claude Code assistance in Cursor IDE.** The context engineering setup ensures Claude Code has all the information it needs to write correct, modern code.

## Setup

### 1. Open Project in Cursor IDE

**Windows:**
```powershell
# From terminal
cursor C:\Users\lefab\Documents\Dev\MidiDrumiGen

# Or from Cursor IDE:
# File → Open Folder → Select MidiDrumiGen
```

### 2. Verify Context is Loaded

In Cursor's chat, test that context documents are accessible:

```
@docs 01_project_overview.md
What is this project about?
```

You should get a response that mentions PyTorch, mido, and NO mention of Magenta or TensorFlow.

## Development Workflow

### Implementing New Features

**Step 1: Load Relevant Context**

In Cursor chat, reference the relevant context documents:

```
@docs 02_architecture.md @docs 04_midi_operations.md
I need to implement MIDI export functionality. Show me the structure.
```

**Step 2: Ask Claude Code to Implement**

```
@docs 04_midi_operations.md @file src/midi/io.py
Implement the export_pattern function in src/midi/export.py following the architecture.
```

Claude Code will:
- Reference the context documents automatically
- Follow patterns from `.cursorrules`
- Use mido (not pretty-midi)
- Write type-hinted Python 3.11 code
- Follow the architecture defined in context docs

**Step 3: Review and Iterate**

Ask Claude Code to:
- Explain the implementation
- Add error handling
- Write tests
- Refactor if needed

### Common Development Tasks

#### Adding a New API Endpoint

```
@docs 02_architecture.md @file src/api/main.py
Add a new endpoint /api/v1/styles that returns available producer styles.
```

#### Implementing MIDI Processing

```
@docs 04_midi_operations.md @docs 03_dependencies.md
Implement humanization functions in src/midi/humanize.py using mido.
```

#### Training Pipeline

```
@docs 05_ml_pipeline.md @file src/models/transformer.py
Create the training script in src/training/train.py following the documented architecture.
```

#### Debugging Issues

```
@docs 06_common_tasks.md
I'm getting a CUDA out of memory error. What should I do?
```

## Using Context Documents

### When to Use Each Document

- **`@docs 01_project_overview.md`** - Starting new features, understanding project scope
- **`@docs 02_architecture.md`** - Designing features, understanding data flow
- **`@docs 03_dependencies.md`** - Adding libraries, checking compatibility
- **`@docs 04_midi_operations.md`** - MIDI-related features
- **`@docs 05_ml_pipeline.md`** - ML/training features
- **`@docs 06_common_tasks.md`** - Quick reference, troubleshooting

### Combining Contexts

You can load multiple documents:

```
@docs 02_architecture.md @docs 04_midi_operations.md @docs 03_dependencies.md
Implement the complete MIDI export pipeline with humanization.
```

### Combining with Code

Reference specific files along with context:

```
@file src/api/main.py @docs 02_architecture.md
Review this API implementation against our architecture.
```

## Running Commands

### Windows PowerShell Commands

All commands should be run in Cursor's integrated terminal (which uses PowerShell on Windows):

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run verification
python scripts/verify_installation.py

# Start API server
uvicorn src.api.main:app --reload

# Start Celery worker (in new terminal)
celery -A src.tasks.worker worker --loglevel=info
```

### Using Cursor's Terminal

1. Open integrated terminal: `` Ctrl+` ``
2. Terminal will use your activated virtual environment
3. Run commands as normal
4. Open multiple terminals for different services (API, Celery, etc.)

## Best Practices

### 1. Always Load Context First

Before implementing, load relevant context documents:

```
@docs 02_architecture.md
Where should I add this new feature?
```

### 2. Let Claude Code Write Code

Don't write code manually - ask Claude Code:

```
@docs 04_midi_operations.md
Implement the validate_drum_pattern function in src/midi/validate.py
```

### 3. Review Generated Code

Ask Claude Code to explain:

```
@file src/midi/export.py
Explain how this export function works and verify it follows our architecture.
```

### 4. Iterate with Context

If something doesn't match, reference context:

```
@docs 02_architecture.md @file src/api/routes/generate.py
This endpoint doesn't match our architecture. Fix it.
```

### 5. Use Context for Debugging

```
@docs 06_common_tasks.md
I'm getting this error: [paste error]. What's the solution?
```

## Verification

### Check Context is Working

Test that Claude Code knows about the project:

```
@docs 03_dependencies.md
What MIDI library should I use?
```

Expected: "mido 1.3.3" (NOT pretty-midi)

```
@docs 01_project_overview.md
What framework should I use for ML?
```

Expected: "PyTorch 2.4+" (NOT TensorFlow)

### Verify Code Quality

Ask Claude Code to check:

```
@file src/models/transformer.py
Review this code for:
1. Type hints
2. Docstrings
3. Architecture compliance
4. Modern Python 3.11 features
```

## Troubleshooting

### Claude Code Doesn't Know About Project

1. Reload Cursor window: `Ctrl+Shift+P` → "Reload Window"
2. Explicitly reference context: `@docs 01_project_overview.md`
3. Check `.cursorcontext/` files exist

### Commands Don't Work

1. Make sure virtual environment is activated
2. Check you're in the project root directory
3. Use PowerShell commands on Windows (not bash)

### Context Not Loading

1. Verify files exist: `Get-ChildItem .cursorcontext\`
2. Check file names match exactly (01_project_overview.md, etc.)
3. Reload Cursor window

## Summary

**Key Points:**

1. ✅ Use Cursor IDE with Claude Code for development
2. ✅ Load context documents with `@docs` before implementing
3. ✅ Let Claude Code write code based on context
4. ✅ Use Windows PowerShell commands in Cursor's terminal
5. ✅ Reference context documents for debugging
6. ✅ Follow `.cursorrules` for coding standards

**Remember:** Claude Code has access to all context documents and will automatically use them when you reference `@docs` or `@file`. You don't need to manually copy code - let Claude Code generate it based on the architecture and patterns defined in the context documents.

