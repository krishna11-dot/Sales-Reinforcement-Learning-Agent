# Quick Reference - Essential Commands

## GitHub Setup (First Time)

```bash
cd c:\Users\krish\Downloads\Sales_Optimization_Agent
git init
git add .
git commit -m "Initial commit: Sales RL Agent implementation"
git remote add origin https://github.com/yourusername/Sales-Reinforcement-Learning-Agent.git
git branch -M main
git push -u origin main
```

Replace `yourusername` with your actual GitHub username.

---

## GitHub Updates (After Changes)

```bash
git add .
git commit -m "Description of changes"
git push
```

---

## Project Structure

```
Sales_Optimization_Agent/
├── README.md              # Main documentation (root)
├── src/                   # Python source code
├── data/                  # Datasets (train/val/test)
├── checkpoints/           # Trained models (.pkl files)
├── logs/                  # Results (.json files)
├── visualizations/        # Training curves (.png files)
└── docs/                  # All other documentation
```

---

## Run Evaluations

```bash
# Activate environment
conda activate Sales_Optimization_Agent

# Evaluate baseline (1.50% performance)
python src/evaluate.py

# Evaluate feature selection (0.80% performance)
python src/evaluate_feature_selection.py

# Analyze features
python src/analyze_features.py
```

---

## Key Results

| Model | Subscription Rate | Improvement | Winner |
|-------|------------------|-------------|---------|
| Baseline | 1.50% | 3.4x | Use This |
| Feature Selection | 0.80% | 1.8x | Don't Use |
| Random | 0.44% | 1.0x | Baseline |

**Recommendation:** Use baseline agent for production.

---

## Documentation Files (in docs/)

- **ARCHITECTURE_SIMPLE.md** - System design and pipeline
- **RESULTS_EXPLAINED.md** - Why feature selection failed
- **COMPLETE_WORKFLOW.md** - All commands and usage
- **FEATURE_SELECTION_DESIGN.md** - Implementation details

---

## Important Notes

1. **README.md stays in root** (GitHub homepage)
2. **All other .md files in docs/** (organized)
3. **Models already trained** (just run evaluations)
4. **Baseline is better** (1.50% vs 0.80%)

---

## File Sizes Warning

Check before pushing to GitHub:
- Model files: `checkpoints/*.pkl` (547 KB - 11 MB)
- Log files: `logs/*.json` (up to 7.5 MB)

If too large, add to `.gitignore` or use Git LFS.

---

## Contact for Issues

- GitHub Issues: https://github.com/yourusername/Sales-Reinforcement-Learning-Agent/issues
- Check GITHUB_SETUP.md for detailed instructions
