# GitHub Clone and Push Guide: Personal to Work Repository

**Purpose:** Transfer code from personal GitHub to work GitHub repository

**Date:** May 7, 2026

---

## Quick Reference

```bash
# Method 1: Clone and add work remote (Recommended)
git clone https://github.com/PERSONAL-USER/REPO.git
cd REPO
git remote add work https://github.boozallencsn.com/Participants/Track2_Team5.git
git remote -v
git push work main

# Method 2: Change remote URL
git clone https://github.com/PERSONAL-USER/REPO.git
cd REPO
git remote set-url origin https://github.com/WORK-ORG/REPO.git
git push origin main


# Configure your work identity for this repo
git config user.email "639250@boozallen.com"
git config user.name "Your Name"
 
# Push to the work repository
git push work main
# (or 'git push work master' if the default branch is master)

# Method 3: Fork on GitHub (Best for long-term)
# Use GitHub web interface to fork, then clone work fork
```

---

## Method 1: Clone and Add Work Remote (Recommended)

### When to Use

- You want to maintain both personal and work remotes
- You need to sync between personal and work repositories
- You want flexibility to push to either repository

### Steps

```bash
# Step 1: Clone from personal GitHub
git clone https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME.git
cd REPO-NAME

# Step 2: Check current remote
git remote -v
# Output:
# origin  https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME.git (fetch)
# origin  https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME.git (push)

# Step 3: Add work GitHub as new remote
git remote add work https://github.com/YOUR-WORK-ORG/REPO-NAME.git

# Step 4: Verify both remotes exist
git remote -v
# Output:
# origin  https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME.git (fetch)
# origin  https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME.git (push)
# work    https://github.com/YOUR-WORK-ORG/REPO-NAME.git (fetch)
# work    https://github.com/YOUR-WORK-ORG/REPO-NAME.git (push)

# Step 5: Configure work identity for this repository
git config user.email "your.email@work.com"
git config user.name "Your Work Name"

# Step 6: Push to work GitHub
git push work main
# Or if your default branch is 'master':
git push work master

# Step 7 (Optional): Push specific branch
git push work statistical-metrics

# Step 8 (Optional): Make work GitHub the default remote
git remote rename origin personal
git remote rename work origin

# Now 'git push' defaults to work GitHub
git push  # Goes to work GitHub
```

### Future Usage

```bash
# Pull from personal
git pull personal main

# Pull from work
git pull origin main  # (if you renamed work to origin)
git pull work main    # (if you kept work as work)

# Push to work
git push origin main  # (if you renamed work to origin)
git push work main    # (if you kept work as work)

# Push to personal
git push personal main
```

---

## Method 2: Change Remote URL

### When to Use

- You only need the work repository going forward
- You don't need to maintain connection to personal repository
- Simplest approach for one-time transfer

### Steps

```bash
# Step 1: Clone from personal GitHub
git clone https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME.git
cd REPO-NAME

# Step 2: Change remote URL to work GitHub
git remote set-url origin https://github.com/YOUR-WORK-ORG/REPO-NAME.git

# Step 3: Verify the change
git remote -v
# Output:
# origin  https://github.com/YOUR-WORK-ORG/REPO-NAME.git (fetch)
# origin  https://github.com/YOUR-WORK-ORG/REPO-NAME.git (push)

# Step 4: Configure work identity
git config user.email "your.email@work.com"
git config user.name "Your Work Name"

# Step 5: Push to work GitHub
git push origin main

# Step 6: Push all branches (optional)
git push origin --all

# Step 7: Push all tags (optional)
git push origin --tags
```

### Future Usage

```bash
# All git commands now use work GitHub
git pull origin main
git push origin main
git push  # Defaults to work GitHub
```

---

## Method 3: Fork on GitHub Web Interface

### When to Use

- Best for long-term maintenance
- You want clean separation between personal and work
- You want to use GitHub's fork features (pull requests, etc.)

### Steps

```bash
# Step 1: Fork on GitHub (Web Interface)
# 1. Go to: https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME
# 2. Click "Fork" button (top right)
# 3. Select your work organization as destination
# 4. Wait for fork to complete

# Step 2: Clone the work fork
git clone https://github.com/YOUR-WORK-ORG/REPO-NAME.git
cd REPO-NAME

# Step 3: Verify remote
git remote -v
# Output:
# origin  https://github.com/YOUR-WORK-ORG/REPO-NAME.git (fetch)
# origin  https://github.com/YOUR-WORK-ORG/REPO-NAME.git (push)

# Step 4: Add personal repo as upstream (optional, for syncing)
git remote add upstream https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME.git

# Step 5: Verify both remotes
git remote -v
# Output:
# origin    https://github.com/YOUR-WORK-ORG/REPO-NAME.git (fetch)
# origin    https://github.com/YOUR-WORK-ORG/REPO-NAME.git (push)
# upstream  https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME.git (fetch)
# upstream  https://github.com/YOUR-PERSONAL-USERNAME/REPO-NAME.git (push)

# Step 6: Configure work identity
git config user.email "your.email@work.com"
git config user.name "Your Work Name"
```

### Syncing Fork with Personal Repository

```bash
# Pull latest changes from personal repository
git fetch upstream
git checkout main
git merge upstream/main

# Push updates to work repository
git push origin main
```

---

## Authentication Methods

### Option A: HTTPS with Personal Access Token (PAT)

#### Creating a Personal Access Token

```bash
# Step 1: Create PAT on GitHub
# 1. Go to: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
# 2. Click "Generate new token (classic)"
# 3. Give it a name: "Work Laptop - TDTF"
# 4. Select scopes:
#    - repo (all)
#    - workflow (if using GitHub Actions)
# 5. Click "Generate token"
# 6. Copy the token (ghp_xxxxxxxxxxxx)

# Step 2: Use PAT when pushing
git push work main
# Username: YOUR-WORK-USERNAME
# Password: ghp_YOUR_PERSONAL_ACCESS_TOKEN

# Step 3: Cache credentials (so you don't have to enter every time)
git config --global credential.helper store
# Next push will save credentials to ~/.git-credentials

# Alternative: Cache for 1 hour
git config --global credential.helper 'cache --timeout=3600'
```

#### Using PAT in Remote URL

```bash
# Include PAT directly in remote URL (less secure, but convenient)
git remote set-url work https://YOUR-USERNAME:ghp_TOKEN@github.com/WORK-ORG/REPO.git

# Or when adding remote:
git remote add work https://YOUR-USERNAME:ghp_TOKEN@github.com/WORK-ORG/REPO.git
```

---

### Option B: SSH Keys (More Secure)

#### Setting Up SSH Keys

```bash
# Step 1: Generate SSH key pair
ssh-keygen -t ed25519 -C "your.email@work.com"
# When prompted for file location, use:
# Windows: C:\Users\639250\.ssh\id_ed25519_work
# Save to: ~/.ssh/id_ed25519_work

# Press Enter for no passphrase (or set one for extra security)

# Step 2: Start SSH agent
# Windows (PowerShell):
Start-Service ssh-agent
ssh-add C:\Users\639250\.ssh\id_ed25519_work

# Linux/Mac:
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_work

# Step 3: Copy public key
# Windows:
type C:\Users\639250\.ssh\id_ed25519_work.pub

# Linux/Mac:
cat ~/.ssh/id_ed25519_work.pub

# Step 4: Add SSH key to work GitHub
# 1. Go to: GitHub → Settings → SSH and GPG keys → New SSH key
# 2. Title: "Work Laptop"
# 3. Paste the public key (starts with ssh-ed25519)
# 4. Click "Add SSH key"

# Step 5: Test SSH connection
ssh -T git@github.com
# Output: Hi USERNAME! You've successfully authenticated...
```

#### Using SSH with Multiple GitHub Accounts

```bash
# Step 1: Create SSH config file
# Windows: C:\Users\639250\.ssh\config
# Linux/Mac: ~/.ssh/config

# Add this content:
Host github.com-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_personal

Host github.com-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work

# Step 2: Clone using SSH with specific host
git clone git@github.com-work:WORK-ORG/REPO-NAME.git

# Step 3: Or change existing remote to SSH
git remote set-url work git@github.com-work:WORK-ORG/REPO-NAME.git

# Step 4: Push using SSH
git push work main
```

---

## Handling Multiple GitHub Accounts

### Configure Git Identity Per Repository

```bash
# Inside your work repository
cd REPO-NAME

# Set work email for this repo only (overrides global config)
git config user.email "your.email@work.com"
git config user.name "Your Work Name"

# Verify
git config user.email
# Output: your.email@work.com

# Check global config (unchanged)
git config --global user.email
# Output: your.personal@email.com
```

### Configure Global vs Local Settings

```bash
# Global settings (apply to all repos)
git config --global user.email "your.personal@email.com"
git config --global user.name "Your Personal Name"

# Local settings (apply to current repo only)
git config user.email "your.email@work.com"
git config user.name "Your Work Name"

# View all settings
git config --list

# View where settings come from
git config --list --show-origin
```

---

## Complete Workflow Examples

### Example 1: TDTF Repository Transfer

```bash
# Scenario: Transfer TDTF from personal to Booz Allen GitHub

# Step 1: Clone from personal GitHub
cd "C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos"
git clone https://github.com/YOUR-PERSONAL/TDTF.git TDTF-work
cd TDTF-work

# Step 2: Configure work identity
git config user.email "639250@boozallen.com"
git config user.name "Your Name"

# Step 3: Add work GitHub remote
git remote add work https://github.com/BOOZ-ALLEN-ORG/TDTF.git

# Step 4: Push main branch to work GitHub
git push work main

# Step 5: Push statistical-metrics branch to work GitHub
git checkout statistical-metrics
git push work statistical-metrics

# Step 6: Set work as default remote
git remote rename origin personal
git remote rename work origin

# Step 7: Set upstream for current branch
git branch --set-upstream-to=origin/statistical-metrics statistical-metrics

# Now 'git push' and 'git pull' default to work GitHub
```

---

### Example 2: Smithsonian Repository Transfer

```bash
# Scenario: Transfer Smithsonian project to work GitHub

# Step 1: Clone from personal
cd "C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian"
git clone https://github.com/YOUR-PERSONAL/smithsonian-project.git
cd smithsonian-project

# Step 2: Configure work identity
git config user.email "639250@boozallen.com"
git config user.name "Your Name"

# Step 3: Change remote to work GitHub
git remote set-url origin https://github.com/BOOZ-ALLEN-ORG/smithsonian-project.git

# Step 4: Push all branches
git push origin --all

# Step 5: Push all tags
git push origin --tags
```

---

### Example 3: Maintaining Both Personal and Work Repos

```bash
# Scenario: Keep personal repo for personal work, work repo for work

# Step 1: Clone from personal
git clone https://github.com/YOUR-PERSONAL/REPO.git REPO-work
cd REPO-work

# Step 2: Add work remote
git remote add work https://github.com/WORK-ORG/REPO.git

# Step 3: Configure work identity
git config user.email "your.email@work.com"

# Step 4: Create work-specific branch
git checkout -b work-main

# Step 5: Push work branch to work GitHub
git push work work-main:main

# Step 6: Set upstream
git branch --set-upstream-to=work/main work-main

# Daily workflow:
# - Work on work-main branch
# - Push to work GitHub: git push work work-main:main
# - Pull from personal: git pull origin main
# - Merge personal changes: git merge origin/main
```

---

## Syncing Between Personal and Work

### Sync Script (Windows PowerShell)

```powershell
# sync-repos.ps1
# Sync personal GitHub changes to work GitHub

$ErrorActionPreference = "Stop"

Write-Host "Fetching from personal GitHub..." -ForegroundColor Cyan
git fetch personal

Write-Host "Fetching from work GitHub..." -ForegroundColor Cyan
git fetch work

Write-Host "Merging personal/main into current branch..." -ForegroundColor Cyan
git merge personal/main

if ($LASTEXITCODE -eq 0) {
    Write-Host "Pushing to work GitHub..." -ForegroundColor Cyan
    git push work main
    Write-Host "Sync complete!" -ForegroundColor Green
} else {
    Write-Host "Merge conflicts detected. Resolve conflicts and run 'git push work main' manually." -ForegroundColor Red
}
```

### Sync Script (Bash)

```bash
#!/bin/bash
# sync-repos.sh
# Sync personal GitHub changes to work GitHub

set -e

echo "Fetching from personal GitHub..."
git fetch personal

echo "Fetching from work GitHub..."
git fetch work

echo "Merging personal/main into current branch..."
git merge personal/main

echo "Pushing to work GitHub..."
git push work main

echo "Sync complete!"
```

---

## Troubleshooting

### Issue 1: Permission Denied (publickey)

```bash
# Problem: SSH authentication failing

# Solution 1: Check SSH key is added to agent
ssh-add -l
# If empty, add key:
ssh-add ~/.ssh/id_ed25519_work

# Solution 2: Test SSH connection
ssh -T git@github.com
# Should see: Hi USERNAME! You've successfully authenticated...

# Solution 3: Use HTTPS instead
git remote set-url work https://github.com/WORK-ORG/REPO.git
```

---

### Issue 2: Repository Already Exists on Work GitHub

```bash
# Problem: Work repo exists with different history

# Solution 1: Force push (CAUTION: overwrites work repo)
git push work main --force

# Solution 2: Merge histories (safer)
git pull work main --allow-unrelated-histories
git push work main

# Solution 3: Delete work repo and recreate
# Use GitHub web interface to delete, then push
```

---

### Issue 3: Different Branch Names

```bash
# Problem: Personal uses 'main', work uses 'master'

# Solution 1: Push personal 'main' to work 'master'
git push work main:master

# Solution 2: Rename your branch
git branch -m main master
git push work master

# Solution 3: Set default branch on GitHub
# Go to: Settings → Branches → Default branch → Change to 'main'
```

---

### Issue 4: Authentication Failed

```bash
# Problem: Username/password not working

# Solution 1: Use Personal Access Token instead of password
# Generate PAT on GitHub, use it as password

# Solution 2: Clear cached credentials
git credential-cache exit
# Or on Windows:
git credential-manager uninstall
git credential-manager install

# Solution 3: Update credential helper
git config --global credential.helper manager-core
```

---

### Issue 5: Large Files or History

```bash
# Problem: Repository too large to push

# Solution 1: Use Git LFS for large files
git lfs install
git lfs track "*.psd"
git lfs track "*.zip"
git add .gitattributes
git commit -m "Add Git LFS"
git push work main

# Solution 2: Clean up history
git filter-branch --tree-filter 'rm -rf large-folder' HEAD
git push work main --force

# Solution 3: Shallow clone (if you don't need full history)
git clone --depth 1 https://github.com/PERSONAL/REPO.git
```

---

## Security Best Practices

### Before Pushing to Work GitHub

```bash
# 1. Check for sensitive data
git log --all --full-history -- "*password*"
git log --all --full-history -- "*secret*"
git log --all --full-history -- "*key*"
git log --all --full-history -- "*.env"

# 2. Review .gitignore
cat .gitignore
# Add sensitive patterns:
echo "*.env" >> .gitignore
echo "secrets/" >> .gitignore
echo "*.key" >> .gitignore

# 3. Check commit messages for sensitive info
git log --oneline

# 4. Remove sensitive data from history (if found)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive/file" \
  --prune-empty --tag-name-filter cat -- --all

# 5. Force push to update history
git push work main --force
```

### Recommended .gitignore for Work

```gitignore
# Sensitive files
*.env
*.key
*.pem
secrets/
credentials/

# Personal configs
.vscode/
.idea/
*.local

# OS files
.DS_Store
Thumbs.db

# Python
__pycache__/
*.pyc
venv/
.env

# Logs
*.log
logs/
```

---

## Quick Command Reference

```bash
# View remotes
git remote -v

# Add remote
git remote add <name> <url>

# Remove remote
git remote remove <name>

# Rename remote
git remote rename <old-name> <new-name>

# Change remote URL
git remote set-url <name> <new-url>

# Fetch from remote
git fetch <remote>

# Pull from remote
git pull <remote> <branch>

# Push to remote
git push <remote> <branch>

# Push all branches
git push <remote> --all

# Push all tags
git push <remote> --tags

# Set upstream branch
git branch --set-upstream-to=<remote>/<branch>

# View branch upstream
git branch -vv
```

---

## Recommended Workflow for Booz Allen Projects

```bash
# Initial Setup (One-time)
cd "C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos"
git clone https://github.com/PERSONAL/PROJECT.git PROJECT-work
cd PROJECT-work
git config user.email "639250@boozallen.com"
git config user.name "Your Name"
git remote add work https://github.com/BOOZ-ALLEN/PROJECT.git
git remote rename origin personal
git remote rename work origin

# Daily Workflow
git pull origin main          # Pull latest from work GitHub
# ... make changes ...
git add .
git commit -m "Description"
git push origin main          # Push to work GitHub

# Sync from personal (if needed)
git pull personal main
git push origin main
```

---

**Document Version:** 1.0  
**Last Updated:** May 7, 2026  
**Author:** Your Name  
**Location:** C:\Users\639250\OneDrive - BOOZ ALLEN HAMILTON\Documents\repos\smithsonian
