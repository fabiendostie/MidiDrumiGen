# Quick Conventional Commit Script for PowerShell
# MidiDrumiGen v2.0

$types = @("feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore")
$scopes = @("api", "research", "llm", "db", "midi", "ui", "tasks", "docs", "config", "deps")

Write-Host "`n=== MidiDrumiGen v2.0 - Quick Commit ===" -ForegroundColor Cyan
Write-Host ""

# Select commit type
Write-Host "Select commit type:" -ForegroundColor Yellow
for ($i = 0; $i -lt $types.Count; $i++) {
    Write-Host "  $($i + 1). $($types[$i])"
}
Write-Host ""

do {
    $typeSelection = Read-Host "Enter number (1-$($types.Count))"
} while ([int]$typeSelection -lt 1 -or [int]$typeSelection -gt $types.Count)

$type = $types[[int]$typeSelection - 1]

# Select scope (optional)
Write-Host "`nSelect scope (optional):" -ForegroundColor Yellow
Write-Host "  0. No scope"
for ($i = 0; $i -lt $scopes.Count; $i++) {
    Write-Host "  $($i + 1). $($scopes[$i])"
}
Write-Host ""

$scopeSelection = Read-Host "Enter number (0-$($scopes.Count), or press Enter for no scope)"
if ([string]::IsNullOrWhiteSpace($scopeSelection) -or $scopeSelection -eq "0") {
    $scope = ""
} else {
    $scope = $scopes[[int]$scopeSelection - 1]
}

# Get commit message
Write-Host "`nCommit details:" -ForegroundColor Yellow
$subject = Read-Host "Short description (required)"

if ([string]::IsNullOrWhiteSpace($subject)) {
    Write-Host "Error: Description cannot be empty" -ForegroundColor Red
    exit 1
}

$body = Read-Host "Detailed description (optional, press Enter to skip)"

# Build commit message
if ($scope) {
    $message = "$type($scope): $subject"
} else {
    $message = "$type: $subject"
}

# Stage all changes
Write-Host "`nStaging changes..." -ForegroundColor Cyan
git add -A

# Show what will be committed
Write-Host "`nFiles to be committed:" -ForegroundColor Yellow
git status --short

# Confirm
Write-Host "`nCommit message: " -NoNewline -ForegroundColor Yellow
Write-Host $message -ForegroundColor Green
if ($body) {
    Write-Host "Body: $body" -ForegroundColor Green
}
Write-Host ""

$confirm = Read-Host "Proceed with commit? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Commit cancelled" -ForegroundColor Yellow
    exit 0
}

# Commit
if ($body) {
    git commit -m $message -m $body
} else {
    git commit -m $message
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed!" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Commit successful" -ForegroundColor Green

# Ask to push
Write-Host "`nPush to remote? (y/n)" -ForegroundColor Yellow
$pushConfirm = Read-Host

if ($pushConfirm -eq "y") {
    $currentBranch = git rev-parse --abbrev-ref HEAD
    Write-Host "Pushing to origin/$currentBranch..." -ForegroundColor Cyan
    git push origin $currentBranch

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Pushed to origin/$currentBranch" -ForegroundColor Green
    } else {
        Write-Host "Push failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "Skipped push. Run manually when ready:" -ForegroundColor Yellow
    $currentBranch = git rev-parse --abbrev-ref HEAD
    Write-Host "  git push origin $currentBranch" -ForegroundColor White
}

Write-Host "`n✓ Done!" -ForegroundColor Cyan
