# Complete Workflow - All Commands

## Full Pipeline Execution

### Option 1: Run Everything from Scratch

```bash
# Navigate to project directory
cd c:\Users\krish\Downloads\Sales_Optimization_Agent

# Activate environment
conda activate Sales_Optimization_Agent

# Step 1: Process raw data (10 seconds)
python src/data_processing.py

# Step 2: Train baseline agent (3 minutes)
python src/train.py

# Step 3: Evaluate baseline (30 seconds)
python src/evaluate.py

# Step 4: Train feature selection agent (28 minutes)
python src/train_feature_selection.py

# Step 5: Evaluate feature selection (1 minute)
python src/evaluate_feature_selection.py

# Step 6: Analyze feature importance (1 minute)
python src/analyze_features.py
```

**Total Time:** ~35 minutes

---

### Option 2: Just Run Evaluations (Already Trained)

```bash
# You are here - models already trained

# Evaluate baseline
python src/evaluate.py

# Evaluate feature selection
python src/evaluate_feature_selection.py

# Analyze features
python src/analyze_features.py
```

**Total Time:** ~3 minutes

---

## Individual Commands Explained

### Command 1: Data Processing
```bash
python src/data_processing.py
```

**What it does:**
- Loads data/raw/SalesCRM.xlsx
- Cleans and engineers features
- Splits into train/val/test (70/15/15)
- Saves to data/processed/

**Outputs:**
- crm_train.csv (7,722 customers)
- crm_val.csv (1,655 customers)
- crm_test.csv (1,655 customers)
- historical_stats.json (normalization)

**When to run:** Once, unless data changes

---

### Command 2: Train Baseline
```bash
python src/train.py
```

**What it does:**
- Initializes 16-dim environment
- Creates Q-Learning agent (6 actions)
- Trains for 100,000 episodes
- Saves checkpoints every 10,000 episodes
- Generates training curves

**Outputs:**
- checkpoints/agent_final.pkl (trained model)
- logs/training_metrics_final.json (history)
- visualizations/training_curves.png (plots)

**Console output every 1000 episodes:**
```
Episode 1,000 / 100,000
TECHNICAL METRICS:
  Avg Reward: 25.34
  Epsilon: 0.37
  Q-table size: 1,234 states

BUSINESS METRICS:
  Subscription Rate: 30.5% (with oversampling)
  First Call Rate: 25.3%
  Improvement: 69.3x
```

**Training time:** 3 minutes

**When to run:** Once, or when changing hyperparameters

---

### Command 3: Evaluate Baseline
```bash
python src/evaluate.py
```

**What it does:**
- Loads trained agent from checkpoints/
- Runs 1000 test episodes (no oversampling)
- Uses greedy policy (no exploration)
- Calculates performance metrics

**Output:**
```
TEST SET RESULTS
Episodes: 1000

BUSINESS METRICS:
  Subscription Rate: 1.50% (baseline: 0.44%)
  First Call Rate: 5.30% (baseline: 4.0%)
  Improvement: 3.4x subscriptions

TECHNICAL METRICS:
  Avg Reward: 10.23
  Avg Steps: 1.00
```

**Saved to:** logs/test_results.json

**When to run:** After training, or to re-check performance

---

### Command 4: Train Feature Selection
```bash
python src/train_feature_selection.py
```

**What it does:**
- Initializes 32-dim environment
- Creates Q-Learning agent (22 actions)
- Trains for 100,000 episodes
- Tracks feature selection metrics

**Outputs:**
- checkpoints/agent_feature_selection_final.pkl
- logs/training_metrics_feature_selection_final.json
- visualizations/training_curves_feature_selection.png

**Console output every 1000 episodes:**
```
Episode 1,000 / 100,000
TECHNICAL METRICS:
  Avg Reward: 93.66
  Epsilon: 0.01
  Q-table size: 17,940 states

BUSINESS METRICS:
  Subscription Rate: 33.6% (with oversampling)
  First Call Rate: 14.4%
  Improvement: 76.4x

FEATURE SELECTION METRICS:
  Avg Feature Toggles: 15.00
  Final Features Selected: 0
```

**Training time:** 28 minutes

**When to run:** Once, or when changing feature selection design

---

### Command 5: Evaluate Feature Selection
```bash
python src/evaluate_feature_selection.py
```

**What it does:**
- Loads trained feature selection agent
- Runs 1000 test episodes
- Tracks feature usage
- Compares with baseline

**Output:**
```
TEST SET RESULTS
Episodes: 1000

BUSINESS METRICS:
  Subscription Rate: 0.80% (baseline: 0.44%)
  First Call Rate: 2.50% (baseline: 4.0%)
  Improvement: 1.8x subscriptions

FEATURE SELECTION METRICS:
  Avg Feature Toggles: 14.71
  Avg Features Selected: 0.11 / 16
  Feature Usage: 0.7%
  Data Collection Savings: 99.3%

TOP 10 MOST SELECTED FEATURES:
  1. Country: 8 times (0.8%)
  2. Had_First_Call: 8 times (0.8%)
  ...

COMPARISON: BASELINE vs FEATURE SELECTION
Subscription Rate (%)    Baseline: 1.50    Feature Selection: 0.80
Improvement Factor       3.41x              1.82x
```

**Saved to:** logs/test_results_feature_selection.json

**When to run:** After training feature selection agent

---

### Command 6: Analyze Features
```bash
python src/analyze_features.py
```

**What it does:**
- Loads feature selection agent
- Runs 1000 test episodes
- Tracks which features selected in successful episodes
- Ranks feature importance

**Output:**
```
RESULTS SUMMARY
Success episodes: 11 (1.1%)
Failure episodes: 989 (98.9%)

FEATURE IMPORTANCE (Success Episodes)
Rank   Feature                  Frequency    Percentage
1      Country_ConvRate         11           100.0%
2      Education_ConvRate       11           100.0%
3      Education                10           90.9%
4      Country                  10           90.9%
5      Days_Since_First_Norm    10           90.9%
...

FEATURE SET SIZE
Average features used (Success):  13.73
Average features used (Failure):  9.70

Insight: Agent learned to use ~13.7 features
         instead of all 16 features
```

**Saved to:** logs/feature_analysis_results.json

**When to run:** After evaluating feature selection agent

---

## Viewing Results

### View JSON Results
```bash
# Baseline results
type logs\test_results.json

# Feature selection results
type logs\test_results_feature_selection.json

# Feature analysis
type logs\feature_analysis_results.json

# Training metrics (large file)
type logs\training_metrics_final.json
```

### View Images
```bash
# Training curves (baseline)
start visualizations\training_curves.png

# Training curves (feature selection)
start visualizations\training_curves_feature_selection.png
```

---

## Inspecting Models

### Load and Inspect Baseline Agent
```python
import pickle
from src.agent import QLearningAgent

# Load agent
agent = QLearningAgent()
agent.load('checkpoints/agent_final.pkl')

# Check stats
print(f"Q-table size: {len(agent.q_table)}")
print(f"Episodes trained: {agent.episodes_trained}")
print(f"Current epsilon: {agent.epsilon}")

# View some Q-values
for state, q_vals in list(agent.q_table.items())[:5]:
    print(f"State: {state}")
    print(f"Q-values: {q_vals}")
    print(f"Best action: {q_vals.argmax()}")
    print()
```

### Load and Inspect Feature Selection Agent
```python
from src.agent_feature_selection import QLearningAgentFeatureSelection

# Load agent
agent = QLearningAgentFeatureSelection()
agent.load('checkpoints/agent_feature_selection_final.pkl')

# Check stats
print(f"Q-table size: {len(agent.q_table)}")
print(f"Episodes trained: {agent.episodes_trained}")

# Much larger Q-table
# 522,619 states vs 1,738 for baseline
```

---

## Testing Individual Components

### Test Environment (Baseline)
```python
from src.environment import CRMSalesFunnelEnv

env = CRMSalesFunnelEnv(
    data_path='data/processed/crm_train.csv',
    stats_path='data/processed/historical_stats.json',
    mode='train'
)

# Run one episode
state, info = env.reset()
print(f"Initial state shape: {state.shape}")
print(f"State values: {state}")

action = 1  # Phone call
next_state, reward, done, truncated, info = env.step(action)
print(f"Reward: {reward}")
print(f"Done: {done}")
print(f"Subscribed: {info.get('subscribed', 0)}")
```

### Test Environment (Feature Selection)
```python
from src.environment_feature_selection import CRMFeatureSelectionEnv

env = CRMFeatureSelectionEnv(
    data_path='data/processed/crm_train.csv',
    stats_path='data/processed/historical_stats.json',
    mode='train'
)

# Run one episode
state, info = env.reset()
print(f"State shape: {state.shape}")  # Should be (32,)
print(f"Feature mask: {state[:16]}")  # First 16 are mask
print(f"Features: {state[16:]}")      # Last 16 are features

# Toggle a feature
action = 0  # Toggle Education
next_state, reward, done, truncated, info = env.step(action)
print(f"Done: {done}")  # Should be False (non-terminal)
print(f"New mask: {next_state[:16]}")

# Take CRM action
action = 17  # Phone call
next_state, reward, done, truncated, info = env.step(action)
print(f"Done: {done}")  # Should be True (terminal)
print(f"Final reward: {reward}")
```

### Test Agent
```python
from src.agent import QLearningAgent
import numpy as np

agent = QLearningAgent(n_actions=6)

# Test state discretization
state = np.array([0.347, 0.856, 0.123, 0.789] + [0.5]*12, dtype=np.float32)
discrete = agent._discretize_state(state)
print(f"Continuous: {state[:4]}")
print(f"Discrete: {discrete[:4]}")

# Test action selection
action = agent.select_action(state, training=True)
print(f"Selected action: {action}")

# Test Q-value update
next_state = np.random.rand(16).astype(np.float32)
agent.update(state, action, 10.0, next_state, False)
print(f"Q-table size after 1 update: {len(agent.q_table)}")
```

---

## Debugging Commands

### Check Data Splits
```python
import pandas as pd

train = pd.read_csv('data/processed/crm_train.csv')
val = pd.read_csv('data/processed/crm_val.csv')
test = pd.read_csv('data/processed/crm_test.csv')

print(f"Train: {len(train)} customers")
print(f"Val: {len(val)} customers")
print(f"Test: {len(test)} customers")
print(f"Total: {len(train) + len(val) + len(test)}")

# Check subscription rates
print(f"\nTrain subscribed: {train['Subscribed'].sum()} ({train['Subscribed'].mean()*100:.2f}%)")
print(f"Val subscribed: {val['Subscribed'].sum()} ({val['Subscribed'].mean()*100:.2f}%)")
print(f"Test subscribed: {test['Subscribed'].sum()} ({test['Subscribed'].mean()*100:.2f}%)")
```

### Check Training Progress
```python
import json

# Load training metrics
with open('logs/training_metrics_final.json', 'r') as f:
    metrics = json.load(f)

# Check final performance
final_subs = metrics['technical']['subscriptions'][-1000:]
print(f"Final 1000 episodes subscription rate: {sum(final_subs)/10:.2f}%")

# Plot subscription rate over time
import matplotlib.pyplot as plt
subs = metrics['technical']['subscriptions']
plt.plot(subs)
plt.xlabel('Episode')
plt.ylabel('Subscription (0 or 1)')
plt.title('Subscription Rate Over Training')
plt.show()
```

### Compare Q-Table Sizes
```python
import pickle

# Baseline
with open('checkpoints/agent_final.pkl', 'rb') as f:
    baseline = pickle.load(f)
print(f"Baseline Q-table: {len(baseline['q_table'])} states")

# Feature selection
with open('checkpoints/agent_feature_selection_final.pkl', 'rb') as f:
    feat_sel = pickle.load(f)
print(f"Feature selection Q-table: {len(feat_sel['q_table'])} states")

# Ratio
ratio = len(feat_sel['q_table']) / len(baseline['q_table'])
print(f"Feature selection is {ratio:.1f}x larger")
```

---

## Performance Benchmarking

### Measure Training Speed
```python
import time
from src.environment import CRMSalesFunnelEnv
from src.agent import QLearningAgent

env = CRMSalesFunnelEnv(
    data_path='data/processed/crm_train.csv',
    stats_path='data/processed/historical_stats.json',
    mode='train'
)
agent = QLearningAgent(n_actions=6)

# Time 1000 episodes
start = time.time()
for ep in range(1000):
    state, info = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, done, truncated, step_info = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
    agent.decay_epsilon()

end = time.time()
print(f"1000 episodes in {end-start:.2f} seconds")
print(f"Episodes per second: {1000/(end-start):.1f}")
```

### Measure Evaluation Speed
```python
import time
from src.environment import CRMSalesFunnelEnv
from src.agent import QLearningAgent

env = CRMSalesFunnelEnv(
    data_path='data/processed/crm_test.csv',
    stats_path='data/processed/historical_stats.json',
    mode='test'
)
agent = QLearningAgent()
agent.load('checkpoints/agent_final.pkl')

# Time 1000 test episodes
start = time.time()
for ep in range(1000):
    state, info = env.reset()
    action = agent.select_action(state, training=False)
    next_state, reward, done, truncated, step_info = env.step(action)

end = time.time()
print(f"1000 evaluations in {end-start:.2f} seconds")
print(f"Evaluations per second: {1000/(end-start):.1f}")
```

---

## Modifying Hyperparameters

### Change Training Episodes
Edit `src/train.py`:
```python
# Line 258
agent, tech_metrics, bus_metrics = train_agent(
    n_episodes=10000,  # Change from 100000 to 10000 for faster testing
    log_interval=1000,
    save_interval=5000
)
```

### Change Learning Rate
Edit `src/agent.py`:
```python
# Line 46
learning_rate=0.05,  # Change from 0.1 to 0.05 for more conservative updates
```

### Change Complexity Penalty
Edit `src/environment_feature_selection.py`:
```python
# Line 293
complexity_penalty = -0.5 * n_active_features  # Change from -0.01 to -0.5
```

### Change Batch Sampling
Edit `src/environment.py`:
```python
# Line 138
if np.random.rand() < 0.5:  # Change from 0.3 to 0.5 (more subscribed samples)
    idx = np.random.choice(self.subscribed_indices)
```

---

## Complete Workflow Summary

### Workflow 1: First-Time Setup
```bash
conda activate Sales_Optimization_Agent
python src/data_processing.py
python src/train.py
python src/evaluate.py
python src/train_feature_selection.py
python src/evaluate_feature_selection.py
python src/analyze_features.py
```

### Workflow 2: Re-Evaluation Only
```bash
conda activate Sales_Optimization_Agent
python src/evaluate.py
python src/evaluate_feature_selection.py
```

### Workflow 3: Re-Train Baseline Only
```bash
conda activate Sales_Optimization_Agent
python src/train.py
python src/evaluate.py
```

### Workflow 4: Re-Train Feature Selection Only
```bash
conda activate Sales_Optimization_Agent
python src/train_feature_selection.py
python src/evaluate_feature_selection.py
python src/analyze_features.py
```

---

## Expected Outputs Summary

After running complete pipeline:

### Files Created:
```
data/processed/
  - crm_train.csv
  - crm_val.csv
  - crm_test.csv
  - historical_stats.json

checkpoints/
  - agent_final.pkl (baseline)
  - agent_feature_selection_final.pkl (feature selection)
  - agent_episode_*.pkl (checkpoints every 10k)

logs/
  - test_results.json (baseline: 1.50%)
  - test_results_feature_selection.json (feature selection: 0.80%)
  - feature_analysis_results.json
  - training_metrics_final.json
  - metrics_episode_*.json

visualizations/
  - training_curves.png (baseline)
  - training_curves_feature_selection.png
```

### Console Outputs:
```
Data processing: Split sizes, feature counts
Training: Episode progress, metrics every 1000 episodes
Evaluation: Final performance metrics, comparison
Analysis: Feature importance rankings
```

---

## Quick Reference

**View baseline results:**
```bash
type logs\test_results.json
```

**View feature selection results:**
```bash
type logs\test_results_feature_selection.json
```

**Run complete pipeline:**
```bash
conda activate Sales_Optimization_Agent && python src/data_processing.py && python src/train.py && python src/evaluate.py && python src/train_feature_selection.py && python src/evaluate_feature_selection.py && python src/analyze_features.py
```

**Re-evaluate everything:**
```bash
python src/evaluate.py && python src/evaluate_feature_selection.py
```

---

## Conclusion

All commands are now documented. You can:
1. Run complete pipeline from scratch (35 min)
2. Re-evaluate trained models (3 min)
3. Inspect individual components
4. Debug specific issues
5. Modify hyperparameters
6. Benchmark performance

Your codebase is complete and ready to run.
