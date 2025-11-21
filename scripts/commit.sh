#!/bin/bash
# Quick Conventional Commit Script for Git Bash/Linux/macOS
# MidiDrumiGen v2.0

# Colors
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}\n=== MidiDrumiGen v2.0 - Quick Commit ===${NC}\n"

# Commit types
types=("feat" "fix" "docs" "style" "refactor" "perf" "test" "build" "ci" "chore")

# Scopes
scopes=("api" "research" "llm" "db" "midi" "ui" "tasks" "docs" "config" "deps")

# Select commit type
echo -e "${YELLOW}Select commit type:${NC}"
select type in "${types[@]}"; do
    if [ -n "$type" ]; then
        break
    fi
done

# Select scope (optional)
echo -e "\n${YELLOW}Select scope (optional, press Enter to skip):${NC}"
echo "0. No scope"
select scope in "${scopes[@]}"; do
    if [ "$REPLY" = "0" ] || [ -z "$REPLY" ]; then
        scope=""
        break
    elif [ -n "$scope" ]; then
        break
    fi
done

# Get commit message
echo -e "\n${YELLOW}Commit details:${NC}"
read -p "Short description (required): " subject

if [ -z "$subject" ]; then
    echo -e "${RED}Error: Description cannot be empty${NC}"
    exit 1
fi

read -p "Detailed description (optional, press Enter to skip): " body

# Build commit message
if [ -n "$scope" ]; then
    message="$type($scope): $subject"
else
    message="$type: $subject"
fi

# Stage all changes
echo -e "\n${CYAN}Staging changes...${NC}"
git add -A

# Show what will be committed
echo -e "\n${YELLOW}Files to be committed:${NC}"
git status --short

# Confirm
echo -e "\n${YELLOW}Commit message:${NC} ${GREEN}$message${NC}"
if [ -n "$body" ]; then
    echo -e "${YELLOW}Body:${NC} ${GREEN}$body${NC}"
fi

read -p $'\n'"Proceed with commit? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo -e "${YELLOW}Commit cancelled${NC}"
    exit 0
fi

# Commit
if [ -n "$body" ]; then
    git commit -m "$message" -m "$body"
else
    git commit -m "$message"
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Commit failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Commit successful${NC}"

# Ask to push
read -p $'\n'"${YELLOW}Push to remote? (y/n): ${NC}" push_confirm

if [ "$push_confirm" = "y" ]; then
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    echo -e "${CYAN}Pushing to origin/$current_branch...${NC}"
    git push origin "$current_branch"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Pushed to origin/$current_branch${NC}"
    else
        echo -e "${RED}Push failed!${NC}"
        exit 1
    fi
else
    current_branch=$(git rev-parse --abbrev-ref HEAD)
    echo -e "${YELLOW}Skipped push. Run manually when ready:${NC}"
    echo -e "  git push origin $current_branch"
fi

echo -e "\n${CYAN}✓ Done!${NC}\n"
