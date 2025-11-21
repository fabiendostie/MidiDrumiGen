# Git Workflow & Conventional Commits

**Version:** 2.0.0
**Last Updated:** 2025-11-17
**Branch Strategy:** main → dev → feature branches

---

## Branch Strategy

### Branch Structure

```
main (production-ready)
  └── dev (integration branch)
       ├── feature/research-pipeline
       ├── feature/llm-providers
       ├── feature/database-layer
       └── feature/max-for-live
```

### Branch Purposes

**`main`** - Production-ready code
- Only accepts merges from `dev`
- Every merge represents a release
- Protected branch (requires PR + review)
- Tagged with version numbers (v2.0.0, v2.1.0, etc.)

**`dev`** - Integration/staging branch
- Main development branch
- Accepts merges from feature branches
- Should always be in a working state
- Continuous testing happens here

**`feature/*`** - Feature branches
- Created from `dev`
- Merged back to `dev` when complete
- Naming: `feature/description-of-feature`
- Examples:
  - `feature/research-orchestrator`
  - `feature/openai-provider`
  - `feature/max-for-live-ui`

**`hotfix/*`** - Emergency fixes
- Created from `main`
- Merged to both `main` and `dev`
- Naming: `hotfix/critical-bug-description`

---

## Conventional Commits

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(llm): add Anthropic provider support` |
| `fix` | Bug fix | `fix(api): resolve CORS error in generate endpoint` |
| `docs` | Documentation only | `docs: update PRD with new requirements` |
| `style` | Code style (formatting, no logic change) | `style: format code with black` |
| `refactor` | Code refactoring | `refactor(research): simplify collector interface` |
| `perf` | Performance improvement | `perf(db): add index on artist_name column` |
| `test` | Add or update tests | `test(api): add integration tests for generation` |
| `build` | Build system or dependencies | `build: update FastAPI to 0.115.4` |
| `ci` | CI/CD changes | `ci: add GitHub Actions workflow` |
| `chore` | Maintenance tasks | `chore: clean up old log files` |
| `revert` | Revert previous commit | `revert: revert "feat: add feature X"` |

### Scopes

Common scopes for this project:

- `api` - FastAPI endpoints
- `research` - Research pipeline components
- `llm` - LLM providers (OpenAI, Claude, Gemini)
- `db` - Database layer
- `midi` - MIDI operations
- `ui` - Max for Live device
- `tasks` - Celery tasks
- `docs` - Documentation
- `config` - Configuration files
- `deps` - Dependencies

### Examples

```bash
# Feature additions
feat(llm): implement Google Gemini provider with fallback logic
feat(research): add audio analysis collector for tempo detection
feat(api): create /research endpoint for artist lookup

# Bug fixes
fix(db): resolve pgvector connection timeout issue
fix(midi): correct velocity humanization algorithm
fix(api): handle missing artist in StyleProfile query

# Documentation
docs(architecture): update system architecture diagram
docs(api): add OpenAPI documentation for all endpoints
docs: create setup guide for new developers

# Refactoring
refactor(research): extract common collector logic to base class
refactor(llm): simplify prompt builder interface

# Performance
perf(db): add composite index on (artist_name, research_status)
perf(research): parallelize collector execution with asyncio

# Tests
test(llm): add unit tests for provider fallback logic
test(api): add integration test for complete research flow

# Build/Dependencies
build(deps): update openai to 1.54.5
build(docker): optimize Docker image size
build: add Python 3.12 support

# Chores
chore: archive old training code to docs/_old
chore(logs): rotate log files older than 30 days
chore: update .gitignore for MIDI cache files
```

### Breaking Changes

For breaking changes, add `BREAKING CHANGE:` in the footer:

```bash
feat(api)!: redesign generate endpoint to accept StyleProfile ID

BREAKING CHANGE: The /generate endpoint now requires a style_profile_id
parameter instead of artist_name. Update all client code to fetch the
profile ID first via /research/{artist} endpoint.

Migration guide: See docs/MIGRATION_v2.md
```

---

## Workflow Commands

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/MidiDrumiGen.git
cd MidiDrumiGen

# Setup main and dev branches
git checkout -b dev
git push -u origin dev

# Configure Git for conventional commits
git config commit.template .gitmessage.txt
```

### Daily Workflow

#### 1. Start New Feature

```bash
# Ensure dev is up to date
git checkout dev
git pull origin dev

# Create feature branch
git checkout -b feature/llm-provider-manager

# Start coding...
```

#### 2. Commit Changes (Frequently!)

```bash
# Stage changes
git add src/generation/providers/manager.py

# Commit with conventional format
git commit -m "feat(llm): implement provider manager with fallback logic

- Add LLMProviderManager class
- Implement automatic fallback to secondary providers
- Add cost tracking per provider
- Include unit tests for fallback scenarios

Closes #42"

# Or use interactive mode
git commit
# Opens editor with template
```

#### 3. Push Frequently

```bash
# Push to remote (creates backup and enables collaboration)
git push origin feature/llm-provider-manager

# Push after every significant commit (recommended: every 1-2 hours)
```

#### 4. Keep Branch Updated

```bash
# Update from dev regularly (daily or more)
git checkout dev
git pull origin dev

git checkout feature/llm-provider-manager
git merge dev

# Or use rebase (cleaner history)
git rebase dev

# Resolve conflicts if any, then
git push origin feature/llm-provider-manager --force-with-lease
```

#### 5. Complete Feature

```bash
# Ensure all tests pass
pytest tests/

# Final commit
git add .
git commit -m "feat(llm): complete provider manager implementation

All tests passing. Ready for review."

# Push
git push origin feature/llm-provider-manager

# Create Pull Request on GitHub
# Target: dev branch
# Reviewers: Assign team members
# Labels: feature, needs-review
```

---

## Quick Commit Script

### commit.sh (Git Bash/Linux/macOS)

```bash
#!/bin/bash
# Quick commit script with conventional commit format

echo "Select commit type:"
select type in "feat" "fix" "docs" "style" "refactor" "perf" "test" "build" "ci" "chore"; do
    break
done

read -p "Scope (e.g., api, llm, db): " scope
read -p "Short description: " subject
read -p "Detailed description (optional, press Enter to skip): " body

if [ -z "$scope" ]; then
    message="$type: $subject"
else
    message="$type($scope): $subject"
fi

if [ -n "$body" ]; then
    git commit -m "$message" -m "$body"
else
    git commit -m "$message"
fi

echo ""
echo "Commit created. Push now? (y/n)"
read -p "> " push

if [ "$push" = "y" ]; then
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    git push origin $current_branch
    echo "✓ Pushed to origin/$current_branch"
fi
```

### commit.ps1 (PowerShell/Windows)

```powershell
# Quick commit script for PowerShell

$types = @("feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore")

Write-Host "Select commit type:"
for ($i = 0; $i -lt $types.Count; $i++) {
    Write-Host "$($i + 1). $($types[$i])"
}

$selection = Read-Host "Enter number (1-$($types.Count))"
$type = $types[[int]$selection - 1]

$scope = Read-Host "Scope (e.g., api, llm, db) or press Enter"
$subject = Read-Host "Short description"
$body = Read-Host "Detailed description (optional, press Enter to skip)"

if ($scope) {
    $message = "$type($scope): $subject"
} else {
    $message = "$type: $subject"
}

if ($body) {
    git commit -m $message -m $body
} else {
    git commit -m $message
}

Write-Host ""
Write-Host "Commit created. Push now? (y/n)"
$push = Read-Host

if ($push -eq "y") {
    $currentBranch = git rev-parse --abbrev-ref HEAD
    git push origin $currentBranch
    Write-Host "✓ Pushed to origin/$currentBranch" -ForegroundColor Green
}
```

### Usage

```bash
# Make executable (Linux/macOS)
chmod +x scripts/commit.sh

# Run
./scripts/commit.sh

# Or add alias to .bashrc/.zshrc
alias gc='./scripts/commit.sh'
```

```powershell
# Windows PowerShell
.\scripts\commit.ps1

# Or add to PowerShell profile
function gc { .\scripts\commit.ps1 }
```

---

## Pre-commit Hooks

### Setup commitlint

```bash
# Install commitlint (Node.js required)
npm install --save-dev @commitlint/{cli,config-conventional}

# Create config
echo "module.exports = {extends: ['@commitlint/config-conventional']}" > commitlint.config.js

# Install Husky
npm install --save-dev husky
npx husky install

# Add commit-msg hook
npx husky add .husky/commit-msg 'npx --no-install commitlint --edit "$1"'
```

### Python Alternative: pre-commit

```bash
# Install pre-commit
pip install pre-commit

# Already configured in .pre-commit-config.yaml
pre-commit install
pre-commit install --hook-type commit-msg
```

### .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 24.10.0
    hooks:
      - id: black

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.4.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
        args: []
```

---

## Auto-Push on Save (VS Code/Cursor)

### .vscode/settings.json

```json
{
  "git.enableSmartCommit": true,
  "git.autofetch": true,
  "git.autorefresh": true,
  "git.confirmSync": false,
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,

  // Optional: Auto-stage on save
  "git.postCommitCommand": "push",

  // Conventional Commits extension settings
  "conventionalCommits.autoCommit": false,
  "conventionalCommits.scopes": [
    "api",
    "research",
    "llm",
    "db",
    "midi",
    "ui",
    "tasks",
    "docs",
    "config",
    "deps"
  ]
}
```

### Recommended VS Code Extensions

```json
{
  "recommendations": [
    "vivaxy.vscode-conventional-commits",
    "eamodio.gitlens",
    "github.vscode-pull-request-github",
    "mhutchie.git-graph"
  ]
}
```

---

## Automated Commit Script (Use Wisely!)

### auto-commit.sh

```bash
#!/bin/bash
# Auto-commit script for rapid development
# Usage: ./auto-commit.sh "working on feature X"

CONTEXT=$1

if [ -z "$CONTEXT" ]; then
    echo "Usage: ./auto-commit.sh <context>"
    exit 1
fi

# Check if there are changes
if [ -z "$(git status --porcelain)" ]; then
    echo "No changes to commit"
    exit 0
fi

# Get current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Determine commit type based on files changed
if git diff --cached --name-only | grep -q "test"; then
    TYPE="test"
elif git diff --cached --name-only | grep -q "docs"; then
    TYPE="docs"
elif git diff --cached --name-only | grep -q ".md$"; then
    TYPE="docs"
else
    TYPE="feat"
fi

# Create timestamp
TIMESTAMP=$(date +"%H:%M")

# Commit message
MESSAGE="$TYPE: WIP - $CONTEXT @ $TIMESTAMP

Auto-committed during active development.
Branch: $BRANCH"

# Stage all changes
git add -A

# Commit
git commit -m "$MESSAGE"

# Push
git push origin $BRANCH

echo "✓ Auto-committed and pushed to $BRANCH"
```

### Usage

```bash
# Quick save during development
./auto-commit.sh "implementing LLM provider manager"

# Add to alias for even faster use
alias save='./auto-commit.sh'

# Then just:
save "adding fallback logic"
```

**⚠️ Warning:** Auto-commits should be squashed before merging to dev!

---

## Squashing WIP Commits

Before creating a PR from your feature branch:

```bash
# Interactive rebase to squash WIP commits
git rebase -i dev

# In the editor, mark commits to squash:
# pick abc1234 feat: start LLM provider manager
# squash def5678 feat: WIP - adding OpenAI provider @ 14:30
# squash ghi9012 feat: WIP - adding Claude provider @ 15:45
# pick jkl3456 test: add provider manager tests

# Save and close editor

# Write proper commit message
# Then force push (safe with --force-with-lease)
git push origin feature/llm-provider-manager --force-with-lease
```

---

## Pull Request Template

### .github/pull_request_template.md

```markdown
## Description
<!-- Brief description of changes -->

## Type of Change
- [ ] feat: New feature
- [ ] fix: Bug fix
- [ ] docs: Documentation
- [ ] refactor: Code refactoring
- [ ] perf: Performance improvement
- [ ] test: Tests
- [ ] build: Build/dependencies
- [ ] ci: CI/CD

## Scope
<!-- e.g., api, llm, research, db -->

## Changes Made
-
-
-

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] All tests passing

## Checklist
- [ ] Code follows conventional commits
- [ ] Documentation updated
- [ ] No linting errors
- [ ] Branch is up to date with dev
- [ ] Ready for review

## Related Issues
Closes #

## Screenshots (if applicable)

```

---

## Git Aliases

Add to `.gitconfig` or run:

```bash
# Conventional commit helpers
git config --global alias.feat '!f() { git commit -m "feat($1): $2"; }; f'
git config --global alias.fix '!f() { git commit -m "fix($1): $2"; }; f'
git config --global alias.docs '!f() { git commit -m "docs: $1"; }; f'

# Quick push
git config --global alias.pushf 'push --force-with-lease'

# Pretty log
git config --global alias.lg "log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"

# Usage:
git feat api "add generate endpoint"
git fix db "resolve connection timeout"
git docs "update architecture diagram"
```

---

## Commit Frequency Guidelines

### ✅ DO Commit:
- After completing a logical unit of work (function, class, feature)
- Before switching tasks
- Before leaving for the day
- After fixing a bug
- After adding tests
- Every 1-2 hours during active development

### ❌ DON'T Commit:
- Broken code (unless using WIP prefix on feature branch)
- Code that doesn't compile
- With uncommitted merge conflicts
- Generated files (unless intentional)
- Secrets or API keys

### Ideal Rhythm

```
09:00 - Start work
10:30 - First commit (feat: implement provider interface)
11:00 - Quick commit (feat: add OpenAI provider)
12:00 - Lunch (commit before leaving)
13:30 - Commit (feat: add Claude provider)
14:00 - Commit (test: add provider tests)
15:30 - Commit (docs: document provider usage)
16:30 - Final commit (feat: complete LLM provider manager)
16:35 - Push and create PR
```

**Aim for: 4-6 commits per day of active feature development**

---

## Branch Protection Rules

### For `main` branch:
- Require pull request reviews (minimum 1)
- Require status checks to pass
- Require branches to be up to date
- No direct pushes
- No force pushes
- Require signed commits (optional but recommended)

### For `dev` branch:
- Require pull request reviews (minimum 1)
- Require status checks to pass
- Allow force pushes with lease
- Require conventional commit format

---

## Versioning (SemVer)

### Version Format: MAJOR.MINOR.PATCH

- **MAJOR**: Breaking changes (v2.0.0 → v3.0.0)
- **MINOR**: New features, backward compatible (v2.0.0 → v2.1.0)
- **PATCH**: Bug fixes, backward compatible (v2.0.1 → v2.0.2)

### Tagging Releases

```bash
# After merging to main
git checkout main
git pull origin main

# Create annotated tag
git tag -a v2.0.0 -m "Release v2.0.0: On-demand research + LLM generation

Major Features:
- Multi-source research pipeline
- OpenAI, Claude, Gemini support
- PostgreSQL with vector search
- Max for Live device

Breaking Changes:
- Removed PyTorch training pipeline
- New API endpoints

Migration guide: docs/MIGRATION_v2.md"

# Push tag
git push origin v2.0.0

# Create GitHub Release from tag
```

---

## CI/CD Integration

Commits trigger automated workflows:

- **feat/fix commits:** Run full test suite
- **docs commits:** Build and deploy documentation
- **PR to dev:** Run tests + linting
- **Merge to main:** Deploy to production + create release

---

## Quick Reference

```bash
# Daily workflow
git checkout dev && git pull
git checkout -b feature/my-feature
# ... make changes ...
git add src/module/file.py
git commit -m "feat(module): add new functionality"
git push origin feature/my-feature

# Quick commit (using script)
./scripts/commit.sh

# Update from dev
git checkout dev && git pull
git checkout feature/my-feature
git merge dev

# Complete feature
git checkout dev
git merge --no-ff feature/my-feature
git push origin dev
git branch -d feature/my-feature
```

---

**Remember:** Good commits tell a story. Your future self (and teammates) will thank you!
