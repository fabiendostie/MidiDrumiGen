# BMAD Commands in Claude Code

## The Problem

BMAD (Brownfield Method for Agentic Development) uses a **nested command structure** with **colon-separated names**:

```
/bmad:bmm:workflows:document-project
/bmad:bmm:agents:analyst
/bmad:core:workflows:brainstorming
```

**Claude Code doesn't support this syntax.** It only recognizes:
- Flat command files in `.claude/commands/`
- Simple naming like `/my-command`
- No support for nested directories or colon syntax

## Current State

Your repository has **70+ BMAD commands** in nested directories:

```
.claude/commands/bmad/
├── bmm/
│   ├── agents/
│   │   ├── analyst.md
│   │   ├── architect.md
│   │   ├── dev.md
│   │   └── ...
│   └── workflows/
│       ├── document-project/
│       ├── architecture/
│       └── ...
├── bmb/
└── core/
```

These **won't show up** in Claude Code's slash command menu because of the nested structure.

## Solution 1: Use Alias Commands (Quick Fix)

I've created **alias commands** in the root `.claude/commands/` directory that reference the full BMAD workflows:

**Available now (restart Claude Code to see them):**
- `/document-project` → Loads BMAD document-project workflow
- `/bmm-analyst` → Loads BMM Analyst agent
- `/bmm-architect` → Loads BMM Architect agent
- `/bmm-dev` → Loads BMM Dev agent

**To use:**
1. Restart Claude Code (Ctrl+R or Cmd+R)
2. Type `/` to see the command menu
3. Select one of the aliases above

The alias will then load the full BMAD workflow from the nested directory.

## Solution 2: Flatten All Commands (Comprehensive)

To make **all 70+ BMAD commands** available in Claude Code, run the flattening script:

```bash
# Preview what will be done (dry run)
python scripts/flatten_bmad_commands.py --dry-run

# Flatten all commands
python scripts/flatten_bmad_commands.py
```

This will create flat command files like:

```
.claude/commands/
├── bmm-document-project.md
├── bmm-analyst.md
├── bmm-architect.md
├── bmm-prd.md
├── bmm-tech-spec.md
├── core-brainstorming.md
└── ... (70+ files)
```

**Benefits:**
- All BMAD commands visible in Claude Code
- Simple naming: `/bmm-analyst`, `/core-brainstorming`
- Original nested files preserved

**Drawback:**
- Creates many files in the root commands directory
- Less organized than nested structure

## Solution 3: Create Custom Aliases (Manual)

For specific BMAD workflows you use frequently, create alias files manually:

```bash
# Example: Create alias for PRD workflow
cat > .claude/commands/bmm-prd.md << 'EOF'
---
description: 'Create Product Requirements Document using BMAD Method'
---

<workflow-load>
Source: @.bmad/bmm/workflows/2-plan-workflows/prd/workflow.yaml
Load the full BMAD PRD workflow and follow its instructions.
</workflow-load>
EOF
```

Then restart Claude Code and use `/bmm-prd`.

## Recommended Approach

**For MidiDrumiGen v2.0, I recommend:**

1. **Use the alias commands I created** for common workflows:
   - `/document-project` (already works!)
   - `/bmm-analyst`
   - `/bmm-architect`
   - `/bmm-dev`

2. **Create aliases as needed** for other workflows you use frequently

3. **Don't flatten all 70+ commands** unless you really need them all in Claude Code

## Why This Happened

BMAD was designed for **Cursor IDE**, which has more flexible command syntax and supports:
- Nested command directories
- Colon-separated command names
- Command namespacing

Claude Code uses a **simpler model**:
- Flat file structure
- One command = one .md file in `.claude/commands/`
- Command name = filename (without .md)

## Testing the Fix

**Right now:**

1. Restart Claude Code (Ctrl+R or Cmd+R)
2. Type `/` in the chat
3. You should see these new commands:
   - `/document-project`
   - `/bmm-analyst`
   - `/bmm-architect`
   - `/bmm-dev`

If they don't appear, check:
- Files exist in `.claude/commands/` (not nested)
- Files have `---` YAML frontmatter with `description`
- Claude Code is fully restarted

## Creating More Aliases

To add more BMAD workflows, create simple alias files:

**Template:**

```markdown
---
description: 'Short description of what this workflow does'
---

<workflow-load>
Source: @.bmad/[path-to-workflow].yaml
Load the full BMAD workflow and follow its instructions.
</workflow-load>
```

**Example - Sprint Planning:**

```bash
cat > .claude/commands/bmm-sprint-planning.md << 'EOF'
---
description: 'Plan sprint with story breakdown and estimation'
---

<workflow-load>
Source: @.bmad/bmm/workflows/4-implementation/sprint-planning/workflow.yaml
Load the BMAD sprint planning workflow.
</workflow-load>
EOF
```

Then restart Claude Code and use `/bmm-sprint-planning`.

## Summary

✅ **What works now:**
- 4 alias commands created: `/document-project`, `/bmm-analyst`, `/bmm-architect`, `/bmm-dev`
- Original BMAD workflows preserved in nested directories
- Easy to add more aliases as needed

❌ **What doesn't work:**
- Nested command syntax like `/bmad:bmm:workflows:document-project`
- Automatic discovery of nested commands

🔧 **What to do:**
1. Restart Claude Code to see the new commands
2. Use `/document-project` to test
3. Create more aliases for workflows you use frequently
4. Run `python scripts/flatten_bmad_commands.py` if you want all commands available

---

**Note:** This is a limitation of Claude Code's command system, not a bug. BMAD's nested structure is more powerful but not compatible with Claude Code's simpler flat model.
