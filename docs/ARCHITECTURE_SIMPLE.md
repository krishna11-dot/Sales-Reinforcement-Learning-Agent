# System Architecture - Complete Pipeline

## High-Level Overview

```
Raw Data -> Data Processing -> RL Environment -> Q-Learning Agent -> Evaluation -> Business Insights
```

---

## Detailed Architecture

### Module 1: Data Processing (Input)
**File:** `src/data_processing.py`

**Purpose:** Convert raw Excel data into clean train/val/test sets

**Input:**
- `data/raw/SalesCRM.xlsx` (11,032 customers)

**Processing Steps:**
1. Load Excel file
2. Feature engineering (encode categories, calculate conversion rates)
3. Normalize continuous features to [0, 1]
4. Temporal split by date (70% train, 15% val, 15% test)
5. Save normalization statistics (mean, std, min, max)

**Output:**
- `data/processed/crm_train.csv` (7,722 customers)
- `data/processed/crm_val.csv` (1,655 customers)
- `data/processed/crm_test.csv` (1,655 customers)
- `data/processed/historical_stats.json` (normalization stats)

**Key Decision:** Temporal split (not random) to prevent data leakage

---

### Module 2A: RL Environment - Baseline (Decision Box)
**File:** `src/environment.py`

**Purpose:** Simulate CRM sales funnel interactions

**State Representation:** 16 dimensions
```
[Education, Country, Stage, Contact_Frequency,
 Days_Since_First, Days_Since_Last, Days_Between,
 Had_First_Call, Had_Survey, Had_Demo, Had_Signup,
 Had_Manager, Status_Active, Stages_Completed,
 Education_ConvRate, Country_ConvRate]
```
All normalized to [0, 1]

**Action Space:** 6 discrete actions
```
0: Send Email
1: Make Phone Call
2: Schedule Demo
3: Send Survey
4: No Action (Wait)
5: Assign Account Manager
```

**Reward Structure:**
- Subscription (terminal): +100
- First call achieved: +15
- Demo scheduled: +10
- Survey sent: +5
- Action cost: -1 per step
- Complexity bonus: -0.1 * num_features (baseline: always -1.6)

**Episode Flow:**
1. Sample random customer from dataset
2. Agent sees 16-dim state
3. Agent selects 1 action (0-5)
4. Episode ends (all actions are terminal)
5. Reward based on customer outcome

**Batch Sampling Strategy:**
- 30% subscribed customers
- 30% first-call customers
- 40% random customers
- Handles 65:1 class imbalance

**Key Design Decision:** Terminal actions (episode ends after 1 step) makes learning faster

---

### Module 2B: RL Environment - Feature Selection (Decision Box)
**File:** `src/environment_feature_selection.py`

**Purpose:** Same as baseline, but allows feature selection

**State Representation:** 32 dimensions
```
[Feature_Mask (16 binary), Customer_Features (16 continuous)]

Feature_Mask: Which features are currently active (0 or 1)
Customer_Features: Same 16 features as baseline
```

**Action Space:** 22 discrete actions
```
0-15: Toggle feature ON/OFF (non-terminal, episode continues)
16: Send Email (terminal, episode ends)
17: Make Phone Call (terminal)
18: Schedule Demo (terminal)
19: Send Survey (terminal)
20: No Action (terminal)
21: Assign Account Manager (terminal)
```

**Reward Structure:**
- Subscription (terminal): +100
- Stage rewards: +15 (call), +10 (demo), +5 (survey)
- Complexity penalty: -0.01 * num_active_features
- Simplicity bonus (terminal): -0.1 * num_active_features
- Action cost: -1 per step

**Episode Flow:**
1. Sample random customer
2. Initialize feature_mask = all 1s (all features ON)
3. Agent sees 32-dim state [mask, features]
4. Agent can:
   - Toggle features (actions 0-15, episode continues)
   - Take CRM action (actions 16-21, episode ends)
5. Reward based on outcome + complexity penalties

**Key Design Decision:** Non-terminal toggles allow exploration, but increase state space

---

### Module 3A: Q-Learning Agent - Baseline (Learning)
**File:** `src/agent.py`

**Purpose:** Learn optimal CRM action for each customer

**Algorithm:** Tabular Q-Learning

**Q-Table Structure:**
```python
Q-table: {state: [Q(s,0), Q(s,1), ..., Q(s,5)]}
State: Tuple of 16 features rounded to 2 decimals
Actions: 6 discrete actions
```

**Learning Rule:**
```
Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

alpha (learning rate): 0.1
gamma (discount factor): 0.95
```

**Exploration Strategy:** Epsilon-greedy
```
epsilon_start: 1.0 (100% random at start)
epsilon_end: 0.01 (1% random at end)
epsilon_decay: 0.995 (exponential decay)
```

**State Discretization:**
- Round each feature to 2 decimal places
- Convert to tuple for dictionary key
- Example: [0.347, 0.856] -> (0.35, 0.86)

**Action Selection:**
- Training: epsilon-greedy (explore vs exploit)
- Evaluation: greedy (always best action)

**Key Design Decision:** 2 decimal discretization balances precision vs sparsity

---

### Module 3B: Q-Learning Agent - Feature Selection (Learning)
**File:** `src/agent_feature_selection.py`

**Purpose:** Learn feature selection + CRM action

**Algorithm:** Tabular Q-Learning (same as baseline)

**Q-Table Structure:**
```python
Q-table: {state: [Q(s,0), Q(s,1), ..., Q(s,21)]}
State: Tuple of 32 features rounded to 2 decimals
Actions: 22 discrete actions (16 toggles + 6 CRM)
```

**Same Learning Parameters:**
- alpha: 0.1
- gamma: 0.95
- epsilon: 1.0 -> 0.01 (decay 0.995)

**Key Difference:** 22 actions instead of 6, 32-dim state instead of 16-dim

**State Space Size:**
- Baseline: ~1,700 unique states
- Feature Selection: ~520,000 unique states (300x larger)

**Key Design Decision:** Same algorithm, larger state/action space

---

### Module 4A: Training Loop - Baseline (Optimization)
**File:** `src/train.py`

**Purpose:** Train agent for 100,000 episodes

**Training Process:**
```
For each episode (1 to 100,000):
    1. Reset environment (sample customer)
    2. Get initial state (16-dim)
    3. Agent selects action (epsilon-greedy)
    4. Environment returns reward
    5. Agent updates Q-table
    6. Decay epsilon
    7. Log metrics every 1000 episodes
    8. Save checkpoint every 10,000 episodes
```

**Metrics Tracked:**
- Technical: rewards, epsilon, Q-table size
- Business: subscription rate, first-call rate
- Progress: episode number, training steps

**Outputs:**
- `checkpoints/agent_final.pkl` (trained agent)
- `logs/training_metrics_final.json` (training history)
- `visualizations/training_curves.png` (performance plots)

**Training Time:** ~3 minutes for 100,000 episodes

**Key Design Decision:** Separate technical vs business metrics for different audiences

---

### Module 4B: Training Loop - Feature Selection (Optimization)
**File:** `src/train_feature_selection.py`

**Purpose:** Same as baseline, with feature selection tracking

**Additional Metrics:**
- Average feature toggles per episode
- Average features selected at decision time
- Feature selection patterns over time

**Training Time:** ~28 minutes for 100,000 episodes (9x slower due to larger state space)

**Outputs:**
- `checkpoints/agent_feature_selection_final.pkl`
- `logs/training_metrics_feature_selection_final.json`
- `visualizations/training_curves_feature_selection.png`

**Key Design Decision:** Track feature selection behavior for analysis

---

### Module 5A: Evaluation - Baseline (Validation)
**File:** `src/evaluate.py`

**Purpose:** Test agent on held-out test set

**Evaluation Process:**
```
For 1000 test episodes:
    1. Sample customer from test set (no oversampling)
    2. Get state
    3. Agent selects action (greedy, no exploration)
    4. Record outcome (subscription, first call, reward)
```

**Metrics Calculated:**
- Subscription rate on test set
- First call rate on test set
- Average reward
- Improvement over baseline (0.44% random)

**Output:**
- `logs/test_results.json`
- Console summary

**Result:** 1.50% subscription rate (3.4x improvement)

**Key Design Decision:** No oversampling in evaluation (realistic performance)

---

### Module 5B: Evaluation - Feature Selection (Validation)
**File:** `src/evaluate_feature_selection.py`

**Purpose:** Test feature selection agent on test set

**Additional Metrics:**
- Average features selected
- Feature usage percentage
- Data collection savings
- Top 10 most selected features

**Output:**
- `logs/test_results_feature_selection.json`
- Comparison with baseline

**Result:** 0.80% subscription rate (1.8x improvement, worse than baseline)

**Key Design Decision:** Compare directly with baseline to show feature selection doesn't help

---

### Module 6: Feature Importance Analysis (Insights)
**File:** `src/analyze_features.py`

**Purpose:** Understand which features matter most

**Analysis Process:**
```
For 1000 test episodes:
    1. Run episode with greedy policy
    2. Track which features were active at decision time
    3. Track outcome (success or failure)
    4. Count feature frequency in successful episodes
```

**Outputs:**
- Feature importance ranking (success episodes)
- Feature importance ranking (failure episodes)
- Average features used (success vs failure)
- Top feature combinations
- `logs/feature_analysis_results.json`

**Key Findings:**
- Country_ConvRate: 100% (always in successful episodes)
- Education_ConvRate: 100%
- Average features in success: 13.7 out of 16

**Key Design Decision:** Separate analysis for success vs failure to identify what matters

---

## Complete Data Flow

### Baseline Pipeline:
```
1. Raw Data (11,032 customers)
   |
   v
2. Data Processing (temporal split, normalization)
   |
   v
3. Train/Val/Test Sets (7,722 / 1,655 / 1,655)
   |
   v
4. RL Environment (16-dim state, 6 actions, batch sampling)
   |
   v
5. Q-Learning Agent (1,738 states, epsilon-greedy)
   |
   v
6. Training Loop (100k episodes, 3 minutes)
   |
   v
7. Evaluation (1000 test episodes, no oversampling)
   |
   v
8. Results: 1.50% subscription rate (3.4x improvement)
```

### Feature Selection Pipeline:
```
1-3. Same as baseline
   |
   v
4. RL Environment (32-dim state, 22 actions)
   |
   v
5. Q-Learning Agent (522,619 states, epsilon-greedy)
   |
   v
6. Training Loop (100k episodes, 28 minutes)
   |
   v
7. Evaluation (1000 test episodes)
   |
   v
8. Results: 0.80% subscription rate (1.8x improvement, WORSE)
   |
   v
9. Feature Analysis: Country & Education most important
```

---

## Module Interactions

```
data_processing.py
    |
    | Outputs: crm_train.csv, crm_val.csv, crm_test.csv, historical_stats.json
    v
environment.py / environment_feature_selection.py
    |
    | Provides: state, action_space, reward, done
    v
agent.py / agent_feature_selection.py
    |
    | Learns: Q-table mapping states to action values
    v
train.py / train_feature_selection.py
    |
    | Runs: 100k episodes, saves checkpoints, logs metrics
    v
evaluate.py / evaluate_feature_selection.py
    |
    | Tests: agent on test set, calculates business metrics
    v
analyze_features.py
    |
    | Extracts: feature importance, optimal feature count
    v
Business Insights
```

---

## Key Design Decisions Summary

### Decision 1: Temporal Data Split
**Why:** Prevent data leakage (future info in training)
**Impact:** Realistic evaluation, lower performance than random split

### Decision 2: Batch Oversampling
**Why:** Handle 65:1 class imbalance
**Strategy:** 30% subscribed, 30% first-call, 40% random
**Impact:** Agent sees enough positive examples to learn

### Decision 3: Reward Shaping
**Why:** Guide learning toward business objective
**Structure:** Large subscription reward (+100), small stage rewards (+5-15), small costs (-1)
**Impact:** Agent prioritizes subscriptions over intermediate stages

### Decision 4: State Discretization (2 decimals)
**Why:** Balance precision vs sparsity
**Trade-off:** 0.35 vs 0.36 treated as different states (lose some generalization)
**Impact:** Manageable Q-table size for baseline (1,738 states)

### Decision 5: Terminal Actions (Baseline)
**Why:** Faster learning, simpler credit assignment
**Trade-off:** Can't learn multi-step strategies
**Impact:** Good performance (1.50%) with simple approach

### Decision 6: Non-Terminal Toggles (Feature Selection)
**Why:** Allow feature exploration
**Trade-off:** Increases state space 300x, harder to learn
**Impact:** Poor performance (0.80%) due to sparse Q-table

---

## Component Responsibilities

### Data Processing Component:
- Load raw data
- Engineer features
- Normalize values
- Split temporally
- Save statistics

### Environment Component:
- Sample customers
- Calculate states
- Execute actions
- Compute rewards
- Handle episodes

### Agent Component:
- Discretize states
- Select actions
- Update Q-values
- Decay exploration
- Save/load models

### Training Component:
- Run episodes
- Track metrics
- Save checkpoints
- Generate plots
- Log progress

### Evaluation Component:
- Test performance
- Calculate metrics
- Compare models
- Save results
- Generate reports

### Analysis Component:
- Extract patterns
- Rank features
- Identify combinations
- Quantify importance
- Provide insights

---

## File Dependencies

```
data_processing.py (no dependencies)
    |
    v
environment.py (depends on: processed data)
    |
    v
agent.py (no dependencies, pure RL algorithm)
    |
    v
train.py (depends on: environment.py, agent.py)
    |
    v
evaluate.py (depends on: environment.py, agent.py, trained model)
    |
    v
analyze_features.py (depends on: environment_feature_selection.py, agent_feature_selection.py, trained model)
```

---

## Execution Order

### First Time Setup:
```
1. python src/data_processing.py          (one time, creates train/val/test)
2. python src/train.py                    (3 min, trains baseline)
3. python src/evaluate.py                 (30 sec, tests baseline)
4. python src/train_feature_selection.py  (28 min, trains feature selection)
5. python src/evaluate_feature_selection.py (1 min, tests feature selection)
6. python src/analyze_features.py         (1 min, analyzes features)
```

### After Models Trained (Re-evaluation):
```
1. python src/evaluate.py                    (test baseline again)
2. python src/evaluate_feature_selection.py  (test feature selection again)
3. python src/analyze_features.py            (analyze features again)
```

---

## System Requirements

### Compute:
- CPU: Any modern processor (no GPU needed)
- RAM: 4 GB minimum (Q-table fits in memory)
- Storage: 2 GB (data + models + logs)

### Software:
- Python 3.10+
- Libraries: numpy, pandas, matplotlib, gymnasium, tqdm

### Time:
- Data processing: 10 seconds
- Baseline training: 3 minutes
- Feature selection training: 28 minutes
- Evaluation: 1 minute each
- Total: ~35 minutes for complete pipeline

---

## Output Summary

### Checkpoints:
- agent_final.pkl (547 KB, baseline model)
- agent_feature_selection_final.pkl (11 MB, feature selection model)

### Logs:
- test_results.json (baseline: 1.50%)
- test_results_feature_selection.json (feature selection: 0.80%)
- feature_analysis_results.json (feature importance)
- training_metrics_final.json (7.5 MB, training history)

### Visualizations:
- training_curves.png (baseline learning progress)
- training_curves_feature_selection.png (feature selection learning progress)

---

## Conclusion

### What Works:
- Baseline Q-Learning: 1.50% (3.4x improvement)
- Data processing pipeline
- Batch oversampling for imbalance
- Temporal data split

### What Doesn't Work:
- Feature selection in state space: 0.80% (worse than baseline)
- 32-dim state space with tabular Q-Learning
- Weak complexity penalty (-0.01)

### Why Feature Selection Failed:
1. State space too large (522K states vs 11K examples)
2. Q-Learning can't generalize
3. All features relevant (no noise to remove)
4. Sparse rewards (1.5% success rate)

### Recommended Approach:
**Use baseline agent for production (1.50% performance, simple, interpretable)**
