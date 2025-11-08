# Cursor IDE Context Engineering Guide

**Complete guide for using Cursor IDE with this project's context documents**

---

## Quick Start

### 1. Open Project in Cursor

```bash
# Open in Cursor IDE
cursor ~/Documents/Dev/MidiDrumiGen

# Or from within the directory:
cd ~/Documents/Dev/MidiDrumiGen
cursor .
```

### 2. Verify Context Files Loaded

Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux), then type:
```
Cursor: Reload Window
```

Your `.cursorcontext/` directory contains 6 documents that Claude Code will reference.

### 3. Use Context References

In the Cursor chat, reference specific documents:

```
@docs 02_architecture.md - Load system architecture
@docs 03_dependencies.md - Load verified dependencies
@folder src/models - Load model implementations
@file src/midi/export.py - Load specific file
```

---

## Context Document Structure

### `.cursorcontext/` Directory

```
.cursorcontext/
├── 01_project_overview.md    - High-level project goals and features
├── 02_architecture.md         - System design and component interactions
├── 03_dependencies.md         - Verified modern libraries (NO legacy)
├── 04_midi_operations.md      - MIDI protocol and drum mapping
├── 05_ml_pipeline.md          - Model training and inference
└── 06_common_tasks.md         - Quick reference for tasks
```

**Total Context:** ~50KB of precise, up-to-date project knowledge

---

## How to Use Each Document

### 1. Starting New Features

**Load:** `@docs 01_project_overview.md`

**When to use:**
- Beginning work on new feature
- Need big picture understanding
- Explaining project to others
- Reviewing project goals

**Example prompts:**
```
@docs 01_project_overview.md
What are the current phase 1 features?

@docs 01_project_overview.md
Help me understand the project's non-goals
```

### 2. System Design Questions

**Load:** `@docs 02_architecture.md`

**When to use:**
- Adding new API endpoints
- Modifying Celery tasks
- Understanding data flow
- Debugging integration issues

**Example prompts:**
```
@docs 02_architecture.md
How does pattern generation flow from API to MIDI export?

@docs 02_architecture.md
Show me the Celery queue configuration
```

### 3. Dependency Issues

**Load:** `@docs 03_dependencies.md`

**When to use:**
- Installing new packages
- Resolving version conflicts
- Checking Python 3.11 compatibility
- Understanding why we DON'T use certain libraries

**Example prompts:**
```
@docs 03_dependencies.md
What's the correct PyTorch installation command?

@docs 03_dependencies.md
Why don't we use pretty-midi?

@docs 03_dependencies.md
What Python version should I use?
```

### 4. MIDI Operations

**Load:** `@docs 04_midi_operations.md`

**When to use:**
- Implementing MIDI export
- Adding humanization features
- Working with GM drum mappings
- Debugging MIDI issues

**Example prompts:**
```
@docs 04_midi_operations.md
How do I apply swing to drum patterns?

@docs 04_midi_operations.md
What are the correct MIDI note numbers for kick and snare?

@docs 04_midi_operations.md
Show me the humanization algorithm
```

### 5. ML Training & Inference

**Load:** `@docs 05_ml_pipeline.md`

**When to use:**
- Training models
- Adding new generation features
- Optimizing inference
- Debugging GPU issues

**Example prompts:**
```
@docs 05_ml_pipeline.md
How do I train the model?

@docs 05_ml_pipeline.md
Show me the model architecture

@docs 05_ml_pipeline.md
How can I reduce GPU memory usage?
```

### 6. Quick Tasks

**Load:** `@docs 06_common_tasks.md`

**When to use:**
- Need quick command reference
- Starting development workflow
- Running tests
- Troubleshooting

**Example prompts:**
```
@docs 06_common_tasks.md
How do I generate a pattern from CLI?

@docs 06_common_tasks.md
What's the command to start Celery workers?

@docs 06_common_tasks.md
How do I add a new producer style?
```

---

## Combining Multiple Contexts

### Multi-Document Queries

Load multiple documents for complex tasks:

```
@docs 02_architecture.md @docs 03_dependencies.md
I want to add a new API endpoint that uses mido. 
Show me the FastAPI pattern and correct mido import.

@docs 04_midi_operations.md @docs 05_ml_pipeline.md
How do I export model-generated tokens to MIDI with humanization?

@docs 01_project_overview.md @docs 02_architecture.md @docs 06_common_tasks.md
Walk me through the complete flow of generating a J Dilla pattern
```

### Code + Context Queries

Combine file references with context docs:

```
@file src/api/routes/generate.py @docs 02_architecture.md
Review this endpoint implementation against our architecture

@folder src/models @docs 05_ml_pipeline.md
Does our model implementation match the documented architecture?

@file src/midi/export.py @docs 04_midi_operations.md
Add swing humanization to this export function
```

---

## Best Practices

### ✅ DO

1. **Start broad, then narrow:**
   ```
   @docs 01_project_overview.md  # Understand project
   @docs 02_architecture.md       # Understand architecture
   @file src/specific/file.py     # Work on specific code
   ```

2. **Reference docs for constraints:**
   ```
   @docs 03_dependencies.md
   What libraries can I use for MIDI parsing?
   # Answer: mido 1.3.3, NOT pretty-midi
   ```

3. **Use docs to prevent mistakes:**
   ```
   @docs 03_dependencies.md
   Should I use TensorFlow for this model?
   # Answer: NO - Use PyTorch 2.4+
   ```

4. **Combine context for complex tasks:**
   ```
   @docs 02_architecture.md @docs 05_ml_pipeline.md @folder src/
   Implement a new Celery task for batch pattern generation
   ```

### ❌ DON'T

1. **Don't assume outdated info:**
   ```
   # WRONG: Can I use Magenta's GrooVAE?
   # RIGHT: @docs 03_dependencies.md Should I use Magenta's GrooVAE?
   ```

2. **Don't skip dependency checks:**
   ```
   # WRONG: pip install some-random-library
   # RIGHT: @docs 03_dependencies.md Is this library compatible?
   ```

3. **Don't ignore architecture:**
   ```
   # WRONG: Create new service without checking architecture
   # RIGHT: @docs 02_architecture.md Where should this component go?
   ```

---

## Context Engineering Workflow

### Adding New Feature

```
Step 1: Understand Feature Scope
@docs 01_project_overview.md
Does this feature align with project goals?

Step 2: Check Architecture
@docs 02_architecture.md
Which component should handle this?

Step 3: Verify Dependencies
@docs 03_dependencies.md
Do I need new dependencies?

Step 4: Review Domain Knowledge
@docs 04_midi_operations.md (if MIDI-related)
OR
@docs 05_ml_pipeline.md (if ML-related)

Step 5: Check Examples
@docs 06_common_tasks.md
Are there similar task examples?

Step 6: Implement
@folder src/relevant_folder @docs 02_architecture.md
Write the code with context loaded
```

### Debugging Issues

```
Step 1: Identify Component
@docs 02_architecture.md
Which component is failing?

Step 2: Check Dependencies
@docs 03_dependencies.md
Are all dependencies correctly installed?

Step 3: Review Implementation
@file failing_file.py @docs 02_architecture.md
Compare implementation to documented patterns

Step 4: Check Common Issues
@docs 06_common_tasks.md
Is this a known issue?
```

---

## Cursor Settings

### Recommended `.cursor/settings.json`

```json
{
  "cursor.chat.modelName": "claude-4.5-sonnet",
  "cursor.chat.contextLength": 200000,
  "cursor.includeComposerInChat": true,
  "cursor.alwaysSearchWeb": false,
  "cursor.autoSuggestEnabled": true,
  "cursor.contextFiles": [
    ".cursorcontext/01_project_overview.md",
    ".cursorcontext/02_architecture.md",
    ".cursorcontext/03_dependencies.md",
    ".cursorcontext/04_midi_operations.md",
    ".cursorcontext/05_ml_pipeline.md",
    ".cursorcontext/06_common_tasks.md"
  ]
}
```

---

## Advanced Usage

### Context Debugging

Check what context Claude Code has:

```
What context documents are currently loaded?

List the dependencies you know about

What's your understanding of the project architecture?
```

### Context Refresh

If Claude seems confused:

1. Reload window: `Cmd+Shift+P` → `Reload Window`
2. Clear cache: `Cmd+Shift+P` → `Clear Cache`
3. Re-reference docs: `@docs 01_project_overview.md`

### Custom Prompts with Context

Save frequently used prompt patterns:

```markdown
## Add New Producer Style Template

@docs 06_common_tasks.md @docs 04_midi_operations.md
I want to add support for [PRODUCER_NAME].

1. What MIDI data do I need?
2. What are the typical style parameters?
3. Walk me through the complete process
```

---

## Verification Checklist

Before starting development, verify:

- [ ] All `.cursorcontext/*.md` files present
- [ ] `.cursorrules` file present
- [ ] Python 3.11 activated
- [ ] Dependencies installed (`pip list`)
- [ ] Context loads correctly in Cursor
- [ ] Can reference `@docs` files

Test with:
```
@docs 03_dependencies.md
What Python version should I use?

Expected answer: Python 3.11
```

---

## Troubleshooting

### Context Not Loading

**Symptom:** Claude doesn't know about project specifics

**Solution:**
1. Check files exist: `ls .cursorcontext/`
2. Reload Cursor: `Cmd+Shift+P` → `Reload Window`
3. Explicitly reference: `@docs 01_project_overview.md`

### Outdated Information

**Symptom:** Claude suggests legacy libraries

**Solution:**
```
@docs 03_dependencies.md
Remind me: Should I use Magenta or PyTorch?

Expected: PyTorch 2.4+, NEVER Magenta
```

### Context Too Large

**Symptom:** Cursor sluggish with all docs loaded

**Solution:**
Load only relevant docs:
```
# Instead of loading everything
@docs 01_project_overview.md @docs 02_architecture.md @docs 03_dependencies.md ...

# Load specific docs
@docs 04_midi_operations.md  # Only MIDI context
```

---

## Pro Tips

### 1. Progressive Context Loading

Start minimal, add as needed:
```
Step 1: @docs 01_project_overview.md (understand project)
Step 2: @docs 06_common_tasks.md (find relevant task)
Step 3: @docs 04_midi_operations.md or @docs 05_ml_pipeline.md (domain knowledge)
Step 4: @file specific_file.py (actual code)
```

### 2. Context Validation

Periodically verify Claude's understanding:
```
@docs 03_dependencies.md
Quick quiz: What's our primary MIDI library?

Expected: mido 1.3.3
```

### 3. Explicit Reminders

When Claude suggests legacy code:
```
STOP. Check @docs 03_dependencies.md
We do NOT use pretty-midi. Use mido instead.
```

### 4. Architecture Enforcement

Before implementing:
```
@docs 02_architecture.md
Where should I add this new feature? Validate against architecture.
```

---

## Summary

**Context documents = AI pair programmer's knowledge base**

- 6 documents cover all project aspects
- ~50KB total size (fits easily in context)
- Always up-to-date with modern stack
- Zero legacy/deprecated information
- Verified dependencies and patterns

**Usage pattern:**
1. Start with overview
2. Check architecture
3. Verify dependencies
4. Reference domain docs
5. Review examples
6. Implement with confidence

**Remember:** These docs prevent Claude from suggesting:
- ❌ Magenta/TensorFlow (legacy)
- ❌ pretty-midi (abandoned)
- ❌ pylive (unverified)
- ❌ Python 3.8/3.9 (old versions)

**They ensure Claude suggests:**
- ✅ PyTorch 2.4+ with CUDA 12.1
- ✅ mido 1.3.3 for MIDI
- ✅ FastAPI + Celery + Redis
- ✅ Python 3.11
- ✅ Modern best practices
