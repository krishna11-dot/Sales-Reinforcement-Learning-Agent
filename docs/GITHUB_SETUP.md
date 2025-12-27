# GitHub Setup Guide

## Step-by-Step Commands to Push to GitHub

### Step 1: Initialize Git Repository

```bash
# Navigate to project directory
cd c:\Users\krish\Downloads\Sales_Optimization_Agent

# Initialize git repository
git init
```

Expected output:
```
Initialized empty Git repository in c:/Users/krish/Downloads/Sales_Optimization_Agent/.git/
```

---

### Step 2: Configure Git (First Time Only)

```bash
# Set your name
git config --global user.name "Your Name"

# Set your email (use your GitHub email)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

---

### Step 3: Add All Files to Staging

```bash
# Add all files
git add .

# Check status
git status
```

Expected output:
```
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   .gitignore
        new file:   LICENSE
        new file:   README.md
        new file:   checkpoints/agent_final.pkl
        new file:   checkpoints/agent_feature_selection_final.pkl
        new file:   data/processed/crm_test.csv
        new file:   data/processed/crm_train.csv
        new file:   data/processed/crm_val.csv
        new file:   data/processed/historical_stats.json
        new file:   data/raw/SalesCRM.xlsx
        new file:   docs/ARCHITECTURE.md
        new file:   docs/ARCHITECTURE_SIMPLE.md
        ... (many more files)
```

---

### Step 4: Create Initial Commit

```bash
# Create commit
git commit -m "Initial commit: Sales RL Agent with baseline and feature selection implementations"
```

Expected output:
```
[main (root-commit) abc1234] Initial commit: Sales RL Agent with baseline and feature selection implementations
 XX files changed, XXXX insertions(+)
 create mode 100644 .gitignore
 create mode 100644 LICENSE
 create mode 100644 README.md
 ... (list of all files)
```

---

### Step 5: Connect to GitHub Repository

```bash
# Add remote repository (replace with your GitHub username)
git remote add origin https://github.com/yourusername/Sales-Reinforcement-Learning-Agent.git

# Verify remote
git remote -v
```

Expected output:
```
origin  https://github.com/yourusername/Sales-Reinforcement-Learning-Agent.git (fetch)
origin  https://github.com/yourusername/Sales-Reinforcement-Learning-Agent.git (push)
```

---

### Step 6: Push to GitHub

```bash
# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

Expected output:
```
Enumerating objects: XX, done.
Counting objects: 100% (XX/XX), done.
Delta compression using up to X threads
Compressing objects: 100% (XX/XX), done.
Writing objects: 100% (XX/XX), XX MiB | XX MiB/s, done.
Total XX (delta XX), reused 0 (delta 0), pack-reused 0
To https://github.com/yourusername/Sales-Reinforcement-Learning-Agent.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

## Complete Command Sequence (Copy-Paste)

```bash
# Navigate to project
cd c:\Users\krish\Downloads\Sales_Optimization_Agent

# Initialize git
git init

# Add all files
git add .

# Create commit
git commit -m "Initial commit: Sales RL Agent with baseline and feature selection implementations"

# Add remote (REPLACE WITH YOUR GITHUB USERNAME)
git remote add origin https://github.com/yourusername/Sales-Reinforcement-Learning-Agent.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

---

## Alternative: Using Personal Access Token (Recommended)

If you get authentication errors, use a Personal Access Token instead of password:

### Generate Token:
1. Go to GitHub.com
2. Click your profile picture → Settings
3. Scroll down → Developer settings
4. Personal access tokens → Tokens (classic)
5. Generate new token (classic)
6. Select scopes: `repo` (full control of private repositories)
7. Generate token and COPY IT (you won't see it again)

### Use Token:
```bash
# When pushing, use token as password
git push -u origin main

# Username: yourusername
# Password: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (your token)
```

### Save Credentials (Optional):
```bash
# Cache credentials for 1 hour
git config --global credential.helper cache

# Or store credentials permanently (less secure)
git config --global credential.helper store
```

---

## After Initial Push

### Making Changes and Pushing

```bash
# Check status
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

---

## Useful Git Commands

### Check Status
```bash
git status
```

### View Commit History
```bash
git log --oneline
```

### View Changes
```bash
# Unstaged changes
git diff

# Staged changes
git diff --staged
```

### Undo Changes
```bash
# Unstage file
git restore --staged <file>

# Discard changes
git restore <file>
```

### Create Branch
```bash
# Create and switch to new branch
git checkout -b feature-branch

# Push branch to GitHub
git push -u origin feature-branch
```

### Switch Branches
```bash
git checkout main
```

### Merge Branch
```bash
# Switch to main
git checkout main

# Merge feature branch
git merge feature-branch

# Push merged changes
git push
```

---

## Checking File Sizes (Important!)

GitHub has file size limits (100 MB per file, 1 GB per repository recommended).

### Check large files:
```bash
# Find files larger than 10 MB
find . -type f -size +10M -exec ls -lh {} \;
```

### If you have large files:

Option 1: Use Git LFS (Large File Storage)
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "*.json"

# Add .gitattributes
git add .gitattributes

# Commit and push
git commit -m "Add Git LFS tracking"
git push
```

Option 2: Don't commit large files
Edit `.gitignore`:
```
# Exclude large model files
checkpoints/*.pkl

# Exclude large log files
logs/*.json
```

Then remove from git:
```bash
git rm --cached checkpoints/*.pkl
git rm --cached logs/*.json
git commit -m "Remove large files"
```

---

## Your Specific Repository

Repository name: `Sales-Reinforcement-Learning-Agent`

### Update commands with your username:

```bash
# Replace 'yourusername' with your actual GitHub username
git remote add origin https://github.com/yourusername/Sales-Reinforcement-Learning-Agent.git
```

Example if your username is 'john123':
```bash
git remote add origin https://github.com/john123/Sales-Reinforcement-Learning-Agent.git
```

---

## File Organization Summary

After moving files to docs/, your structure is:

```
Sales_Optimization_Agent/
├── .git/                  # Git repository (created by git init)
├── .gitignore             # Files to ignore
├── LICENSE                # MIT License
├── README.md              # Main project README (stays in root)
├── GITHUB_SETUP.md        # This file
├── pyproject.toml         # Project configuration
├── requirements.txt       # Python dependencies
│
├── src/                   # Source code
├── data/                  # Data files
├── checkpoints/           # Trained models
├── logs/                  # Results
├── visualizations/        # Plots
└── docs/                  # Documentation (all other .md files)
    ├── ARCHITECTURE.md
    ├── ARCHITECTURE_SIMPLE.md
    ├── FEATURE_SELECTION_DESIGN.md
    ├── RESULTS_EXPLAINED.md
    ├── COMPLETE_WORKFLOW.md
    ├── INSIGHTS_EXPLAINED_SIMPLE.md
    ├── FINAL_SUMMARY.md
    ├── COMMANDS_TO_RUN.md
    ├── WHERE_IS_EVERYTHING.md
    ├── IMPLEMENTATION_COMPLETE.md
    └── README_UPDATED.md
```

---

## Troubleshooting

### Error: "fatal: not a git repository"
Solution: Run `git init` first

### Error: "remote origin already exists"
Solution: Remove and re-add
```bash
git remote remove origin
git remote add origin https://github.com/yourusername/Sales-Reinforcement-Learning-Agent.git
```

### Error: "failed to push some refs"
Solution: Pull first, then push
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Error: "Large files detected"
Solution: Use Git LFS or add to .gitignore (see above)

---

## Verification

After pushing, verify on GitHub:

1. Go to https://github.com/yourusername/Sales-Reinforcement-Learning-Agent
2. Check that all files are there
3. Verify README.md displays correctly on homepage
4. Check docs/ folder has all documentation

---

## Next Steps After Pushing

1. Add repository description on GitHub
2. Add topics/tags: `reinforcement-learning`, `q-learning`, `crm`, `sales-optimization`, `machine-learning`
3. Enable GitHub Pages (optional) to host documentation
4. Add project to your GitHub profile
5. Share repository URL

---

## Summary

**Quick Push (for first time):**
```bash
cd c:\Users\krish\Downloads\Sales_Optimization_Agent
git init
git add .
git commit -m "Initial commit: Sales RL Agent with baseline and feature selection implementations"
git remote add origin https://github.com/yourusername/Sales-Reinforcement-Learning-Agent.git
git branch -M main
git push -u origin main
```

**Regular Updates:**
```bash
git add .
git commit -m "Description of changes"
git push
```

That's it! Your project is now on GitHub.
