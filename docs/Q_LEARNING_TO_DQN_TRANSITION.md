# From Q-Learning to DQN - Understanding the Transition

## The Truth: Did Your Q-Learning Fail?

### Short Answer: YES, But Only For Feature Selection

| Variant | State Space | Q-Table Size | Performance | Status |
|---------|-------------|--------------|-------------|--------|
| **Baseline** | 15 features | 1,449 states | 1.30% (3.0x) | ✅ **SUCCESS** |
| **Feature Selection** | 30 features + binary mask | 522,619 states | 0.80% (1.8x) | ❌ **FAILED** |

### Why Feature Selection Failed

**The State Space Explosion Problem:**

```
Baseline (15 features):
- Possible states: ∞ (continuous)
- Visited states: 1,449
- Data per state: 11,032 / 1,449 = ~7.6 samples per state
- Result: ✅ Sufficient data to learn

Feature Selection (30 features + mask):
- Possible states: ∞ (continuous) × 2^15 (binary masks) = Even more infinite!
- Visited states: 522,619
- Data per state: 11,032 / 522,619 = ~0.021 samples per state
- Result: ❌ Almost every state seen once or never!
```

**Mathematical Reality:**

```
Tabular Q-Learning needs: ~10-20 samples per state to learn

Baseline:
7.6 samples/state ✅ Enough!

Feature Selection:
0.021 samples/state ❌ 99.8% of states never seen twice!
```

**Why This Matters:**

When the Q-table is sparse (most states never visited), the agent:
1. Can't learn patterns (no data)
2. Can't generalize (each state is unique)
3. Defaults to random actions (Q-values stay at 0)
4. Performance drops to baseline or worse

---

## Your Modular Architecture: What Gets Replaced?

### The Three-Module Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    MODULE 1: INPUT                          │
│                 (Data Processing Pipeline)                  │
├─────────────────────────────────────────────────────────────┤
│  • Load raw data (SalesCRM.xlsx)                            │
│  • Feature engineering                                      │
│  • Normalize to [0, 1]                                      │
│  • Split: train/val/test                                    │
│                                                             │
│  Files: data_processing.py                                 │
│  Output: CSV files + historical_stats.json                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              MODULE 2: DECISION BOX (RL)                    │
│              ⚠️ THIS IS WHAT CHANGES ⚠️                     │
├─────────────────────────────────────────────────────────────┤
│  CURRENT (Q-Learning):                                      │
│  ┌─────────────────────────────────────────────┐           │
│  │  environment.py (Gymnasium)                 │           │
│  │  - State: 15-dim continuous vector          │  ✅ KEEP  │
│  │  - Actions: 6 discrete actions              │           │
│  │  - Rewards: +100, +15, -1, etc.             │           │
│  │  - Batch sampling: 30-30-40                 │           │
│  └─────────────────────────────────────────────┘           │
│                       ↓                                     │
│  ┌─────────────────────────────────────────────┐           │
│  │  agent.py (Q-Learning)                      │           │
│  │  - Q-table: dict {state → Q-values}         │  ❌ REPLACE│
│  │  - Discretization: round to 2 decimals      │           │
│  │  - Update: Bellman equation                 │           │
│  │  - Epsilon-greedy exploration               │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
│  NEW (DQN with Stable-Baselines3):                         │
│  ┌─────────────────────────────────────────────┐           │
│  │  environment.py (Gymnasium)                 │           │
│  │  - UNCHANGED! Still same env!               │  ✅ KEEP  │
│  └─────────────────────────────────────────────┘           │
│                       ↓                                     │
│  ┌─────────────────────────────────────────────┐           │
│  │  Stable-Baselines3 (DQN)                    │           │
│  │  - Neural network: input → hidden → Q-vals  │  ✅ NEW   │
│  │  - Experience replay buffer                 │           │
│  │  - Target network                           │           │
│  │  - Automatic epsilon decay                  │           │
│  └─────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 MODULE 3: OUTPUT                            │
│                 (Evaluation & Metrics)                      │
├─────────────────────────────────────────────────────────────┤
│  • Load trained agent                                       │
│  • Test on held-out set                                     │
│  • Calculate metrics                                        │
│  • Generate insights                                        │
│                                                             │
│  Files: evaluate.py, visualizations                        │
│  Output: Metrics, plots, business insights                 │
└─────────────────────────────────────────────────────────────┘
```

### What EXACTLY Gets Replaced?

| Component | Keep or Replace? | Reason |
|-----------|-----------------|--------|
| **data_processing.py** | ✅ **KEEP** | Data pipeline stays same |
| **environment.py** | ✅ **KEEP** | Gymnasium env is universal (works with any RL algorithm) |
| **agent.py** | ❌ **REPLACE** | Q-table → Neural network |
| **train.py** | ⚠️ **MODIFY** | Same loop structure, different API calls |
| **evaluate.py** | ⚠️ **MODIFY** | Same evaluation logic, different model loading |

---

## Q-Learning vs DQN: The Fundamental Difference

### Analogy: Phone Book vs Calculator

**Q-Learning (Tabular) = Phone Book**

```
You have a physical book:
State                              | Q(Email) | Q(Call) | Q(Demo) | ...
-----------------------------------|----------|---------|---------|----
(0.87, 0.45, 3, 0.6, ...)          |   -5.2   |  23.5   |   8.3   | ...
(0.23, 0.12, 1, 0.2, ...)          |   12.1   |  -2.3   |  -5.0   | ...
(0.65, 0.38, 2, 0.4, ...)          |    ?     |    ?    |    ?    | ... ← Never seen!
```

**What happens for new state (0.65, 0.38, 2, 0.4)?**
- Look it up in phone book
- **Not found!** (never visited this exact state)
- Q-values = [0, 0, 0, 0, 0, 0] (defaults)
- Pick random action

**Problem:** Can't generalize. State (0.65, 0.38, 2, 0.4) is probably similar to (0.87, 0.45, 3, 0.6), but Q-table doesn't know that!

---

**DQN (Neural Network) = Calculator**

```
You have a function (neural network):
Input: State = (0.87, 0.45, 3, 0.6, ...)
       ↓
   [Neural Network]
   (learns patterns)
       ↓
Output: Q-values = [-5.2, 23.5, 8.3, 2.1, -1.0, 5.6]
```

**What happens for new state (0.65, 0.38, 2, 0.4)?**
- Feed into neural network
- Network thinks: "Hmm, this is similar to states I've seen before"
- **Calculates** Q-values: [-3.1, 18.2, 6.5, 1.8, -0.5, 4.2]
- Picks best action: Call (Q=18.2)

**Advantage:** Generalizes! Learns that "high education + medium country → Call works well"

---

### Visual Comparison

**Q-Learning (Lookup Table):**

```
State Space (all possible states)
┌─────────────────────────────────────────────┐
│  •    •         •    •                      │
│      •    •    •         •        •         │  Possible states: INFINITE
│  •        •       •              •    •     │
│        ●    ●    ●              •           │  Visited states: 1,449 (●)
│  •    •    ●       •      •          •      │  Unvisited: Everything else (•)
│      •        •         •    •         •    │
│  •         •      •              •     •    │
└─────────────────────────────────────────────┘

For each ●: Store Q-values in table
For each •: Q-values = 0 (never learned)

Generalization: NONE
```

**DQN (Neural Network Function):**

```
State Space (all possible states)
┌─────────────────────────────────────────────┐
│  ≈    ≈         ≈    ≈                      │
│      ≈    ≈    ≈         ≈        ≈         │  Possible states: INFINITE
│  ≈        ≈       ≈              ≈    ≈     │
│        ●    ●    ●              ≈           │  Trained on: 1,449 states (●)
│  ≈    ≈    ●       ≈      ≈          ≈      │  Generalizes to: All states (≈)
│      ≈        ≈         ≈    ≈         ≈    │
│  ≈         ≈      ≈              ≈     ≈    │
└─────────────────────────────────────────────┘

Neural network learns a FUNCTION: f(state) → Q-values
Works for ANY state, even unseen ones!

Generalization: FULL
```

---

## The DQN Solution: Step-by-Step Understanding

### Step 1: What is DQN?

**DQN = Deep Q-Network = Q-Learning + Neural Network**

**Components:**

```
1. Neural Network (replaces Q-table)
   Input: State (15 features)
   Hidden Layers: Learn patterns
   Output: Q-values (6 actions)

2. Experience Replay Buffer
   Stores: (state, action, reward, next_state)
   Purpose: Break correlation, stabilize learning

3. Target Network
   A copy of the main network
   Purpose: Stabilize Q-value targets

4. Epsilon-Greedy (same as Q-Learning)
   Purpose: Exploration vs exploitation
```

### Step 2: How DQN Replaces Q-Table

**Your Current Q-Learning:**

```python
# agent.py (current)
class QLearningAgent:
    def __init__(self):
        # Q-table: dictionary
        self.q_table = defaultdict(lambda: np.zeros(6))

    def select_action(self, state):
        state_key = self._discretize_state(state)  # Round to 2 decimals
        q_values = self.q_table[state_key]         # Lookup in table
        return np.argmax(q_values)                 # Best action

    def update(self, state, action, reward, next_state, done):
        state_key = self._discretize_state(state)
        next_state_key = self._discretize_state(next_state)

        # Bellman update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        target = reward + 0.95 * max_next_q

        self.q_table[state_key][action] += 0.1 * (target - current_q)
```

**DQN (Stable-Baselines3):**

```python
# train_dqn.py (new)
from stable_baselines3 import DQN
from environment import CRMSalesFunnelEnv

# Same environment!
env = CRMSalesFunnelEnv(
    data_path='data/processed/crm_train.csv',
    stats_path='data/processed/historical_stats.json',
    mode='train'
)

# Create DQN agent (replaces QLearningAgent)
model = DQN(
    "MlpPolicy",              # Multi-layer perceptron (neural network)
    env,
    learning_rate=0.0001,     # Like your alpha=0.1, but smaller for NN
    buffer_size=10000,        # Experience replay buffer
    learning_starts=1000,     # Start learning after 1000 steps
    batch_size=32,            # Train on 32 samples at a time
    gamma=0.95,               # Same as your discount factor
    exploration_fraction=0.3, # Like your epsilon decay
    exploration_final_eps=0.01,
    verbose=1
)

# Train (similar to your loop)
model.learn(total_timesteps=100000)

# Save
model.save("checkpoints/dqn_agent")
```

### Step 3: The Neural Network Architecture

**What's Inside the "MlpPolicy"?**

```
Input Layer (15 neurons)
    ↓
[0.87, 0.45, 3, 0.6, 0.34, 0.15, 0.98, 0.12, 1.0, 0.0, 1.0, 0.0, 1.0, 0.01, 2.0]
    ↓
Hidden Layer 1 (64 neurons)
[Activation: ReLU]
    ↓
Hidden Layer 2 (64 neurons)
[Activation: ReLU]
    ↓
Output Layer (6 neurons)
[-5.2, 23.5, 8.3, 2.1, -1.0, 5.6]
  ↓     ↓     ↓     ↓     ↓     ↓
Email  Call  Demo Survey Wait Manager
```

**Key Insight:** This is a FUNCTION, not a lookup!

```python
# Q-Learning (lookup)
q_values = q_table[state]  # O(1) time, but needs exact state in table

# DQN (calculation)
q_values = neural_network(state)  # O(n) time, but works for ANY state
```

---

## Step-by-Step Transition Guide

### Phase 1: Understand Your Current System ✅ (You're Here!)

**What you have:**

```
✅ Gymnasium environment (environment.py)
✅ Tabular Q-Learning agent (agent.py)
✅ Training loop (train.py)
✅ Evaluation script (evaluate.py)
✅ Batch-level balancing (30-30-40)
```

**Performance:**
- Baseline: 1.30% (3.0x improvement) ← Works!
- Feature Selection: 0.80% (1.8x improvement) ← Failed due to state space explosion

---

### Phase 2: Install Stable-Baselines3

```bash
conda activate sales_rl
pip install stable-baselines3[extra]
```

**What you get:**
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- SAC (Soft Actor-Critic)
- Pre-built, optimized, tested implementations

---

### Phase 3: Minimal DQN Implementation

**Create:** `src/train_dqn.py`

```python
"""
DQN Training Script - Replaces Q-Learning Agent

WHAT CHANGED:
- agent.py (Q-table) → DQN (neural network)
- Everything else stays the same!
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from environment import CRMSalesFunnelEnv
import numpy as np

# MODULE 1: INPUT (UNCHANGED)
# Data already processed by data_processing.py

# MODULE 2: DECISION BOX (CHANGED - DQN instead of Q-Learning)

# Create environment (same as before!)
env = CRMSalesFunnelEnv(
    data_path='data/processed/crm_train.csv',
    stats_path='data/processed/historical_stats.json',
    mode='train'
)

# Create DQN agent (replaces QLearningAgent)
model = DQN(
    policy="MlpPolicy",           # Neural network policy
    env=env,
    learning_rate=0.0001,         # Learning rate (like alpha)
    buffer_size=10000,            # Experience replay buffer
    learning_starts=1000,         # Warm-up period
    batch_size=32,                # Mini-batch size
    gamma=0.95,                   # Discount factor (same as yours)
    exploration_fraction=0.3,     # Epsilon decay (30% of training)
    exploration_initial_eps=1.0,  # Start epsilon
    exploration_final_eps=0.01,   # End epsilon
    target_update_interval=1000,  # Update target network frequency
    verbose=1,
    tensorboard_log="./logs/tensorboard/"
)

# Train
print("Training DQN agent...")
model.learn(
    total_timesteps=100000,       # Same as your 100k episodes
    log_interval=1000,            # Log every 1000 steps
    progress_bar=True
)

# Save
model.save("checkpoints/dqn_agent_final")
print("Training complete!")

# MODULE 3: OUTPUT (EVALUATION)

# Evaluate
print("\nEvaluating on test set...")
test_env = CRMSalesFunnelEnv(
    data_path='data/processed/crm_test.csv',
    stats_path='data/processed/historical_stats.json',
    mode='test'
)

mean_reward, std_reward = evaluate_policy(
    model,
    test_env,
    n_eval_episodes=1000,
    deterministic=True
)

print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
```

**Run:**

```bash
python src/train_dqn.py
```

---

### Phase 4: Compare Results

| Metric | Q-Learning (Baseline) | DQN | Expected Improvement |
|--------|----------------------|-----|---------------------|
| **State Space Handling** | 1,449 states | Continuous (infinite) | ✅ Handles any state |
| **Generalization** | None (lookup only) | Full (learned function) | ✅ Better on unseen states |
| **Training Time** | 3 minutes | 10-15 minutes | ⚠️ Slower (neural network) |
| **Subscription Rate** | 1.30% | 1.5-2.0% (expected) | ✅ Should improve |
| **Feature Selection** | 0.80% (failed) | 1.2-1.5% (should work) | ✅ Handles 500k+ states |

---

## Nuances & Concepts - Crystal Clear

### Nuance 1: State Discretization

**Q-Learning (Your Current Code):**

```python
def _discretize_state(self, state):
    """Round to 2 decimals to create discrete key."""
    return tuple(np.round(state, 2))

# Example:
state = [0.87234, 0.45678, 3.12345, ...]
discrete = (0.87, 0.46, 3.12, ...)  # Bucket
q_values = q_table[(0.87, 0.46, 3.12, ...)]  # Lookup
```

**Problem:** States (0.87, 0.46, 3.12) and (0.88, 0.46, 3.12) are treated as COMPLETELY DIFFERENT!

**DQN (No Discretization Needed):**

```python
# No discretization! Feed continuous state directly
state = [0.87234, 0.45678, 3.12345, ...]
q_values = neural_network(state)  # Computes Q-values

# States (0.87, 0.46, 3.12) and (0.88, 0.46, 3.12) are treated as SIMILAR!
# Network learns: "states close together → similar Q-values"
```

---

### Nuance 2: Experience Replay

**Q-Learning (Your Current Code):**

```python
# Learn immediately from current experience
state, action, reward, next_state = experience
agent.update(state, action, reward, next_state, done)  # Update right away
```

**Problem:** Learns from correlated sequences (episode 1000 is all similar customers)

**DQN (Experience Replay):**

```python
# Store experience in buffer
buffer.add(state, action, reward, next_state, done)

# Later, sample RANDOM batch
batch = buffer.sample(32)  # 32 random experiences from history

# Learn from random batch (breaks correlation!)
for experience in batch:
    agent.update(experience)
```

**Benefit:** More stable learning, better data efficiency

---

### Nuance 3: Target Network

**Q-Learning (Your Current Code):**

```python
# Update using same Q-values you're updating!
target = reward + gamma * np.max(q_table[next_state])  # ← Using current Q-table
q_table[state][action] += alpha * (target - q_table[state][action])  # ← Updating same Q-table
```

**Problem:** "Chasing a moving target" - unstable!

**DQN (Target Network):**

```python
# Two networks:
main_network = Neural_Network()     # Updates every step
target_network = Neural_Network()   # Copy of main, updates every 1000 steps

# Update using STABLE target network
target = reward + gamma * target_network(next_state).max()  # ← Using old network
main_network.update(state, action, target)                  # ← Updating main network

# Every 1000 steps:
target_network.copy_weights_from(main_network)  # Sync networks
```

**Benefit:** More stable Q-value targets, faster convergence

---

### Nuance 4: When to Use What?

| Scenario | Algorithm | Reason |
|----------|-----------|--------|
| **Small state space** (<10k states) | Q-Learning | Simple, fast, interpretable |
| **Medium state space** (10k-100k states) | DQN | Handles complexity, generalizes |
| **Large state space** (>100k states) | DQN or PPO | Only option that works |
| **Continuous actions** | PPO or SAC | Q-Learning/DQN only for discrete |
| **Need interpretability** | Q-Learning | Can inspect Q-table |
| **Need best performance** | DQN/PPO | State-of-the-art |

**Your Case:**

- Baseline (1,449 states): Q-Learning works! ✅
- Feature Selection (522k states): Need DQN ✅

---

## The Architecture - What Actually Changes?

### Before (Q-Learning):

```
┌──────────────────────┐
│  data_processing.py  │  ← Stays
└──────────────────────┘
           ↓
┌──────────────────────┐
│   environment.py     │  ← Stays
│   (Gymnasium Env)    │
└──────────────────────┘
           ↓
┌──────────────────────┐
│      agent.py        │  ← REPLACED
│   (Q-table lookup)   │
└──────────────────────┘
           ↓
┌──────────────────────┐
│      train.py        │  ← Modified API calls
└──────────────────────┘
           ↓
┌──────────────────────┐
│    evaluate.py       │  ← Modified API calls
└──────────────────────┘
```

### After (DQN):

```
┌──────────────────────┐
│  data_processing.py  │  ← SAME
└──────────────────────┘
           ↓
┌──────────────────────┐
│   environment.py     │  ← SAME (Gymnasium is universal!)
│   (Gymnasium Env)    │
└──────────────────────┘
           ↓
┌──────────────────────┐
│ Stable-Baselines3    │  ← NEW (replaces agent.py)
│   DQN(MlpPolicy)     │
│  - Neural network    │
│  - Replay buffer     │
│  - Target network    │
└──────────────────────┘
           ↓
┌──────────────────────┐
│   train_dqn.py       │  ← NEW (simpler than old train.py)
└──────────────────────┘
           ↓
┌──────────────────────┐
│  evaluate_dqn.py     │  ← NEW (simpler than old evaluate.py)
└──────────────────────┘
```

**Key Insight:** Your environment is the "test track" - it works with ANY algorithm!

---

## Summary: Your Transition Path

### What Failed?

✅ **Baseline Q-Learning:** 1,449 states → Works great (1.30%)
❌ **Feature Selection Q-Learning:** 522,619 states → Failed (0.80%)

**Why?** Tabular Q-Learning can't handle 500k+ states (too sparse, no generalization)

---

### What Gets Replaced?

| Module | Status | Details |
|--------|--------|---------|
| **Input Module** | ✅ Keep | data_processing.py unchanged |
| **Environment** | ✅ Keep | environment.py unchanged (Gymnasium is universal) |
| **Decision Box** | ❌ Replace | agent.py (Q-table) → Stable-Baselines3 (DQN neural network) |
| **Training Loop** | ⚠️ Modify | train.py → train_dqn.py (simpler API) |
| **Evaluation** | ⚠️ Modify | evaluate.py → evaluate_dqn.py (simpler API) |

---

### Key Concepts

1. **Q-Table vs Neural Network**
   - Q-Table: Phone book (lookup only)
   - Neural Network: Calculator (computes for any input)

2. **Generalization**
   - Q-Learning: None (each state independent)
   - DQN: Full (learns patterns, works on unseen states)

3. **State Space**
   - Q-Learning: Works for <10k states
   - DQN: Works for infinite states (continuous)

4. **Three DQN Enhancements**
   - Neural network (generalization)
   - Experience replay (stability)
   - Target network (stable targets)

---

### Next Steps

1. **Install:** `pip install stable-baselines3[extra]`
2. **Create:** `src/train_dqn.py` (minimal implementation above)
3. **Run:** `python src/train_dqn.py`
4. **Compare:** DQN vs Q-Learning results
5. **Scale:** Try feature selection again with DQN

**Expected Result:** DQN should handle feature selection (522k states) much better than Q-Learning!
