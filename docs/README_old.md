# CRM Sales Pipeline RL Optimization

Reinforcement Learning system to optimize sales actions for CRM pipeline, targeting subscription conversion rate improvement from 0.44% to 1%+.

## Project Overview

**Goal**: Learn optimal sales team actions to maximize subscription conversions while minimizing acquisition costs.

**Approach**: Q-Learning agent with batch-level oversampling for extreme class imbalance (228:1).

**Key Features**:
- Temporal data handling (no leakage)
- Batch-level oversampling (30/30/40)
- Separated technical and business metrics
- 16-dimensional state space
- 6 discrete actions

## Quick Start

### 1. Environment Setup

```bash
# Navigate to project directory
cd Sales_Optimization_Agent

# Create UV virtual environment with Python 3.10.11
uv venv --python 3.10.11

# Activate environment
# Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install gymnasium numpy pandas openpyxl matplotlib seaborn jupyter tqdm scikit-learn
```

### 2. Data Processing

```bash
# Ensure SalesCRM.xlsx is in data/raw/
python src/data_processing.py
```

This creates:
- `data/processed/crm_train.csv`
- `data/processed/crm_val.csv`
- `data/processed/crm_test.csv`
- `data/processed/historical_stats.json`

### 3. Training

```bash
python src/train.py
```

Training runs for 100,000 episodes (~30-90 minutes). Progress logged every 1,000 episodes.

### 4. Evaluation

```bash
python src/evaluate.py
```

Evaluates trained agent on test set.

## Project Structure

```
Sales_Optimization_Agent/
├── data/
│   ├── raw/                    # Original SalesCRM.xlsx
│   └── processed/              # Train/val/test CSVs
├── src/
│   ├── data_processing.py      # Temporal-aware preprocessing
│   ├── environment.py          # Gymnasium environment
│   ├── agent.py                # Q-Learning agent
│   ├── train.py                # Training loop
│   └── evaluate.py             # Evaluation script
├── checkpoints/                # Saved models
├── logs/                       # Training metrics
├── visualizations/             # Training curves
├── pyproject.toml              # UV dependencies
└── README.md
```

## Key Design Decisions

### 1. Temporal Split (No Leakage)
- Split by DATE first (70/15/15)
- Calculate statistics ONLY on train set
- Map train statistics to val/test

### 2. Batch Oversampling (30/30/40)
- 30% subscribed customers
- 30% first call customers
- 40% random (mostly negatives)
- Addresses 228:1 class imbalance

### 3. Reward Shaping
- Terminal: +100 (subscription)
- Intermediate: +15 (first call), +12 (demo), etc.
- Action costs: -1 to -20
- All intermediate < 25% of terminal

### 4. Hyperparameters
- Learning rate: 0.1 (conservative for noisy data)
- Discount factor: 0.95 (multi-step task)
- Epsilon decay: 0.995 (reaches 0.01 at ~1000 episodes)

## Success Criteria

### Technical Metrics
- Q-value convergence: Delta < 0.001
- Action stability: < 5% change
- Q-table size: 5,000-10,000 states

### Business Metrics
- Subscription rate: 0.44% to 1.0%+ (2.3x improvement)
- First call rate: 4.0% to 8.0%+ (2x improvement)
- ROI: Positive (revenue > costs)
- Statistical significance: p < 0.05

## Results

Training results will show:
- Subscription conversion improvements
- Cost per acquisition reduction
- High-value customer segments (Education x Country x Stage)
- Effective action sequences

See `logs/` for detailed metrics and `visualizations/` for training curves.

## Interview Preparation

This project demonstrates:
1. Handling temporal data without leakage
2. Addressing extreme class imbalance
3. Reward shaping for sparse rewards
4. State discretization trade-offs
5. Hyperparameter justification
6. Modular debugging approach
7. Separating technical vs business metrics

See `problem_definition.md` for comprehensive interview prep notes.

## Dependencies

- Python 3.10.11
- Gymnasium (RL environment)
- NumPy (numerical operations)
- Pandas (data processing)
- OpenPyXL (Excel reading)
- Matplotlib/Seaborn (visualization)
- Scikit-learn (metrics)
- TQDM (progress bars)

All managed via UV package manager.

## License

Educational project for interview preparation.
