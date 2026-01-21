# Next Algorithms After Q-Learning

## Introduction

You've successfully built a **Tabular Q-Learning** agent that achieves 3.0x improvement over random baseline (1.30% vs 0.44% subscription rate). This is a great start for reinforcement learning!

However, Q-Learning has limitations - particularly with large state spaces. This document explains what to learn next and why.

---

## Quick Summary Table

| Algorithm | Type | Best For | Complexity | When to Use |
|-----------|------|----------|------------|-------------|
| **Q-Learning** (Current) | Value-based, Tabular | Small state spaces (<10K states) | Low | ✅ You're here! |
| **DQN** | Value-based, Deep | Medium state spaces (10K-1M states) | Medium | Feature selection (30-dim) |
| **Double DQN** | Value-based, Deep | Same as DQN, but more stable | Medium | After mastering DQN |
| **Dueling DQN** | Value-based, Deep | When state value matters separately | Medium | After mastering DQN |
| **PPO** | Policy-based, Deep | Continuous actions, complex policies | High | Enterprise CRM (multi-objective) |
| **A2C/A3C** | Actor-Critic, Deep | Faster training with parallel workers | High | Large-scale production |
| **SAC** | Actor-Critic, Off-Policy | Continuous actions, exploration-exploitation | High | Advanced projects |

---

## 1. Deep Q-Networks (DQN)

### What Is DQN?

**Core idea:** Replace Q-table with a neural network

```python
# Q-Learning (Current):
Q = {}  # Dictionary: {state: [Q-values for 6 actions]}
# Problem: Need to store every state explicitly
# Limitation: 1.4K states OK, 522K states NOT OK

# DQN (Neural Network):
Q_network = NeuralNetwork(input_dim=15, output_dim=6)
Q_values = Q_network(state)  # Predicts Q-values for all actions
# Advantage: Generalizes to unseen states
# Can handle: Millions of states
```

---

### When to Use DQN

**Your current project:**

| Scenario | States | Current Method | Recommended |
|----------|--------|----------------|-------------|
| **Baseline (15-dim)** | 1,449 | ✅ Q-Learning works | Keep Q-Learning |
| **Feature Selection (30-dim)** | 522,619 | ❌ Q-Learning fails (0.80%) | Use DQN |
| **One-Hot Education (45-dim)** | ~1M+ | ❌ Q-Learning impossible | Use DQN |

**Why DQN would help feature selection:**
```
Current problem:
- 30-dim state → 522K states
- 11K training samples
- Result: Most states never visited → weak Q-values

DQN solution:
- Neural network learns: "State [0.5, 0.3, ...] is similar to [0.52, 0.29, ...]"
- Generalizes across similar states
- Doesn't need to visit every state explicitly
- Expected result: 1.5-2.0% (better than 0.80%, maybe better than baseline 1.30%)
```

---

### How DQN Works

**Key components:**

1. **Q-Network:** Neural network that approximates Q(s, a)
2. **Experience Replay:** Store past experiences, sample randomly for training
3. **Target Network:** Separate network for stable Q-value targets
4. **Loss Function:** Mean squared error between predicted and target Q-values

**Architecture:**

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim=30, action_dim=21, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)  # Returns Q-values for all actions

# Usage:
state = torch.tensor([0.5, 0.3, 0.8, ...])  # 30 dimensions
q_values = dqn_model(state)  # [21 Q-values]
action = q_values.argmax()  # Choose action with highest Q-value
```

**Training loop:**

```python
# 1. Collect experience
state = env.reset()
action = select_action(state, epsilon)  # Epsilon-greedy
next_state, reward, done = env.step(action)
replay_buffer.add(state, action, reward, next_state, done)

# 2. Sample batch from replay buffer
batch = replay_buffer.sample(batch_size=64)

# 3. Compute Q-values
current_q = q_network(batch.states)[batch.actions]  # Q(s, a)

# 4. Compute targets (using target network for stability)
next_q = target_network(batch.next_states).max(dim=1)  # max_a' Q(s', a')
target_q = batch.rewards + gamma * next_q * (1 - batch.dones)

# 5. Update Q-network
loss = F.mse_loss(current_q, target_q)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# 6. Update target network periodically
if step % target_update_freq == 0:
    target_network.load_state_dict(q_network.state_dict())
```

---

### Implementation for Your Project

**Option 1: Stable-Baselines3 (Recommended)**

```python
# Install
pip install stable-baselines3

# Train DQN for feature selection
from stable_baselines3 import DQN
from src.environment_feature_selection import CRMEnvironmentFeatureSelection

# Create environment
env = CRMEnvironmentFeatureSelection(train_df, historical_stats)

# Create DQN agent
model = DQN(
    policy="MlpPolicy",  # Multi-layer perceptron
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=0.005,  # Target network update rate
    gamma=0.95,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    verbose=1
)

# Train
model.learn(total_timesteps=100000)

# Save
model.save("checkpoints/dqn_feature_selection")

# Evaluate
obs = env.reset()
total_reward = 0
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        obs = env.reset()

print(f"Subscription rate: {total_reward / 1000}")
```

**Expected timeline:**
- Setup: 30 minutes
- Training: 10-15 minutes (100K steps)
- Evaluation: 2 minutes
- Total: ~1 hour

**Expected results:**
- Subscription rate: 1.5-2.0% (vs 0.80% with tabular Q-Learning)
- Might beat baseline 1.30% (feature selection can now work properly)

---

**Option 2: From Scratch (Learning)**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, learning_rate=1e-4, gamma=0.95):
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def train_step(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and backprop
        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Training loop
agent = DQNAgent(state_dim=30, action_dim=21)
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

for episode in range(10000):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, info = env.step(action)

        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.train_step(batch_size=64)

        state = next_state
        total_reward += reward

    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if episode % 100 == 0:
        agent.update_target_network()
        print(f"Episode {episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")
```

---

### DQN Improvements

**1. Double DQN (DDQN)**

**Problem with DQN:**
```
DQN uses: target = reward + gamma * max_a Q_target(next_state, a)
Issue: max_a overestimates Q-values (always picks highest, even if noise)
```

**Solution:**
```python
# Double DQN: Use online network to SELECT action, target network to EVALUATE
next_actions = q_network(next_states).argmax(dim=1)  # Select with online
next_q = target_network(next_states).gather(1, next_actions.unsqueeze(1))  # Evaluate with target
target_q = rewards + gamma * next_q.squeeze(1) * (1 - dones)
```

**When to use:** After basic DQN works, add this for more stable training

---

**2. Dueling DQN**

**Idea:** Separate state value V(s) from action advantages A(s, a)

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
```

**When to use:** When some states are inherently good/bad regardless of action (e.g., "customer from B27" is valuable, "customer from B1" is not)

---

**3. Prioritized Experience Replay (PER)**

**Idea:** Sample important experiences more often

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done, td_error=1.0):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(abs(td_error) + 1e-5)  # Priority = |TD error|

    def sample(self, batch_size, beta=0.4):
        # Sample according to priorities
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),
            indices
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5
```

**When to use:** When you have rare important events (like subscriptions) that should be learned from more

---

## 2. Policy Gradient Methods

### Proximal Policy Optimization (PPO)

**Core difference from Q-Learning:**

```
Q-Learning (Value-based):
- Learns Q(s, a): "How good is action a in state s?"
- Policy: Choose action with highest Q-value (implicit)
- Works for: Discrete actions

PPO (Policy-based):
- Learns π(a|s): "Probability of action a in state s"
- Policy: Sample action from probability distribution (explicit)
- Works for: Discrete AND continuous actions
```

**When to use PPO:**

| Scenario | Q-Learning/DQN | PPO |
|----------|----------------|-----|
| **Discrete actions (6 actions)** | ✅ Works well | ✅ Also works |
| **Continuous actions (e.g., "wait 3.5 days")** | ❌ Can't handle | ✅ Works well |
| **Multi-objective (subscription + satisfaction)** | ❌ Single reward | ✅ Can balance |
| **Stochastic policy needed** | ❌ Deterministic | ✅ Stochastic |

---

**Your current project:**
- Actions are discrete (Email, Call, Demo, Survey, Wait, Manager)
- Single objective (subscription rate)
- → **Q-Learning/DQN is sufficient**

**When you'd need PPO:**
- Enterprise CRM with continuous actions: "Send email in X days" where X ∈ [0, 7]
- Multi-objective: Maximize subscriptions AND customer satisfaction simultaneously
- Exploration important: Need stochastic policy to try different action sequences

---

**PPO Implementation (if needed):**

```python
from stable_baselines3 import PPO

# Same environment as before
env = CRMEnvironmentFeatureSelection(train_df, historical_stats)

# Create PPO agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,  # Steps per update
    batch_size=64,
    n_epochs=10,  # Epochs per update
    gamma=0.95,
    gae_lambda=0.95,  # Generalized Advantage Estimation
    clip_range=0.2,  # PPO clip parameter
    verbose=1
)

# Train
model.learn(total_timesteps=100000)

# Evaluate
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

**Expected results:**
- Similar to DQN for discrete actions (1.5-2.0%)
- Possibly slightly better exploration
- Longer training time (2-3x slower than DQN)

---

## 3. Actor-Critic Methods

### A2C (Advantage Actor-Critic) / A3C (Asynchronous A3C)

**Core idea:** Combine value-based and policy-based

```
Actor: Learns policy π(a|s) (what action to take)
Critic: Learns value V(s) (how good is this state)

Advantage: A(s, a) = Q(s, a) - V(s)
           = "How much better is action a compared to average?"

Actor update: Maximize advantage (take better-than-average actions more often)
Critic update: Minimize value prediction error (learn accurate state values)
```

**When to use:**
- Faster training with parallel workers (A3C)
- More stable than pure policy gradient (PPO is usually better though)
- Good for continuous control

**Your project:** Probably not needed - PPO is more modern and stable

---

### SAC (Soft Actor-Critic)

**Core idea:** Actor-Critic + Maximum Entropy

```
Standard RL: Maximize reward
SAC: Maximize reward + entropy

Entropy = "How random is my policy?"
→ Encourages exploration naturally
→ Learns robust policies
```

**When to use:**
- Continuous actions (very good for robotics)
- Need strong exploration
- Production systems (robust to environment changes)

**Your project:** Overkill - actions are discrete

---

## 4. Model-Based RL

### World Models, MuZero, etc.

**Core idea:** Learn a model of the environment, then plan

```
Model-free (Q-Learning, DQN, PPO):
- Trial and error in real environment
- Learns directly from experience
- Sample inefficient (needs lots of data)

Model-based:
- Learn P(s'|s, a) (how environment transitions)
- Simulate future trajectories
- Plan optimal actions
- Sample efficient (reuse model)
```

**Your project:** Not suitable
- Environment is already a simulator (not real world)
- No benefit to learning another model on top
- Would need model of customer behavior (complex)

**When useful:**
- Expensive real-world interactions (robotics, autonomous driving)
- Want to simulate "what if?" scenarios
- Have complex dynamics to learn

---

## 5. Multi-Agent RL

### Scenario: Multiple Sales Reps

**If your CRM had:**
- 10 sales reps
- Each rep handles different customers
- Reps can transfer customers to each other
- Goal: Maximize total subscriptions across all reps

**Then:**
```
Single-agent RL (current): Each rep acts independently
Multi-agent RL: Reps coordinate and communicate

Algorithms:
- MADDPG (Multi-Agent DDPG)
- QMIX (Learn joint Q-values)
- CommNet (Communication networks)
```

**Your project:** Not needed - single agent (CRM system) making decisions

---

## Recommendation for Your Project

### Immediate Next Step: DQN for Feature Selection

**Why:**
- Your feature selection FAILED with tabular Q-Learning (0.80% vs 1.30% baseline)
- Root cause: 30-dim state → 522K states → sparse Q-table
- DQN can handle 30-dim state easily → expected 1.5-2.0%

**Implementation plan:**

```
Step 1: Install Stable-Baselines3 (5 minutes)
pip install stable-baselines3

Step 2: Modify train_feature_selection.py (15 minutes)
- Replace tabular Q-Learning with DQN
- Use Stable-Baselines3 API (simple)

Step 3: Train (10-15 minutes)
python src/train_feature_selection_dqn.py

Step 4: Evaluate (2 minutes)
python src/evaluate_feature_selection_dqn.py

Step 5: Compare results
- Tabular Q-Learning: 0.80%
- DQN: 1.5-2.0% (expected)
- Baseline: 1.30%

Step 6: Write up results
- DQN enables feature selection to work
- Learned which features matter most
- Shows understanding of deep RL
```

**Total time:** ~1-2 hours

**Payoff:**
- Demonstrates you can use deep RL (not just tabular)
- Fixes feature selection experiment
- Strong interview talking point
- Might improve performance beyond baseline

---

### Long-Term Learning Path

**After DQN:**

1. **Double DQN + Dueling DQN** (1-2 days)
   - Read papers
   - Implement improvements
   - Compare stability/performance

2. **PPO** (3-5 days)
   - Learn policy gradient theory
   - Implement for discrete actions
   - Understand when PPO > DQN

3. **Advanced Topics** (weeks/months)
   - Multi-objective RL (subscription + satisfaction)
   - Continuous action spaces (wait time as continuous)
   - Offline RL (learn from historical data without exploration)
   - Contextual bandits (simpler than RL, might be sufficient)

---

## Code Example: DQN for Feature Selection

**File: `src/train_feature_selection_dqn.py`**

```python
"""
Train DQN agent for CRM Feature Selection

This solves the state space explosion problem from tabular Q-Learning:
- Tabular: 30-dim state → 522K states → 0.80% performance
- DQN: 30-dim state → Neural network → Expected 1.5-2.0% performance
"""

import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from src.environment_feature_selection import CRMEnvironmentFeatureSelection

# Load data
train_df = pd.read_csv('data/processed/crm_train.csv')
val_df = pd.read_csv('data/processed/crm_val.csv')

# Calculate historical stats on training set only
historical_stats = {
    'country_conv': train_df.groupby('Country')['Subscribed_Binary'].mean().to_dict(),
    'edu_conv': train_df.groupby('Education')['Subscribed_Binary'].mean().to_dict()
}

# Create environments
train_env = CRMEnvironmentFeatureSelection(train_df, historical_stats)
val_env = CRMEnvironmentFeatureSelection(val_df, historical_stats)

# Create DQN agent
print("Creating DQN agent...")
model = DQN(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,  # Start learning after 1000 steps
    batch_size=64,
    tau=0.005,  # Soft update coefficient
    gamma=0.95,
    train_freq=4,  # Train every 4 steps
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.5,  # Explore for 50% of training
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    policy_kwargs=dict(net_arch=[128, 128]),  # 2 hidden layers, 128 units each
    verbose=1,
    tensorboard_log="./logs/tensorboard/"
)

# Create evaluation callback (test on validation set every 10K steps)
eval_callback = EvalCallback(
    val_env,
    best_model_save_path='./checkpoints/dqn_feature_selection_best/',
    log_path='./logs/dqn_feature_selection/',
    eval_freq=10000,
    deterministic=True,
    render=False
)

# Train
print("\nTraining DQN agent...")
print(f"Total timesteps: 100,000")
print(f"Expected training time: 10-15 minutes")
model.learn(
    total_timesteps=100000,
    callback=eval_callback,
    log_interval=1000
)

# Save final model
model.save("checkpoints/dqn_feature_selection_final")
print("\nTraining complete! Model saved to checkpoints/dqn_feature_selection_final.zip")

# Quick evaluation on training set
print("\nQuick evaluation on training set...")
obs = train_env.reset()
total_subscriptions = 0
total_episodes = 0

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = train_env.step(action)
    if done:
        if info.get('subscribed', False):
            total_subscriptions += 1
        total_episodes += 1
        obs = train_env.reset()

train_sub_rate = (total_subscriptions / total_episodes) * 100 if total_episodes > 0 else 0
print(f"Training subscription rate: {train_sub_rate:.2f}%")
print(f"\nRun 'python src/evaluate_feature_selection_dqn.py' to test on held-out test set")
```

**File: `src/evaluate_feature_selection_dqn.py`**

```python
"""
Evaluate DQN agent on test set
"""

import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from src.environment_feature_selection import CRMEnvironmentFeatureSelection

# Load data
train_df = pd.read_csv('data/processed/crm_train.csv')
test_df = pd.read_csv('data/processed/crm_test.csv')

# Historical stats from training set
historical_stats = {
    'country_conv': train_df.groupby('Country')['Subscribed_Binary'].mean().to_dict(),
    'edu_conv': train_df.groupby('Education')['Subscribed_Binary'].mean().to_dict()
}

# Create test environment
test_env = CRMEnvironmentFeatureSelection(test_df, historical_stats)

# Load trained model
print("Loading DQN model from checkpoints/dqn_feature_selection_final.zip...")
model = DQN.load("checkpoints/dqn_feature_selection_final")

# Evaluate
print("\nEvaluating on test set (1000 episodes)...")
obs = test_env.reset()
total_subscriptions = 0
total_episodes = 0
total_actions = 0

feature_usage = np.zeros(15)  # Track which features are used most

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)

    # Track feature toggles
    if action < 15:  # Toggle action
        feature_usage[action] += 1

    obs, reward, done, info = test_env.step(action)
    total_actions += 1

    if done:
        if info.get('subscribed', False):
            total_subscriptions += 1
        total_episodes += 1
        obs = test_env.reset()

# Results
test_sub_rate = (total_subscriptions / total_episodes) * 100 if total_episodes > 0 else 0
actions_per_episode = total_actions / total_episodes if total_episodes > 0 else 0

print("\n" + "="*50)
print("DQN FEATURE SELECTION RESULTS")
print("="*50)
print(f"Test Episodes: {total_episodes}")
print(f"Subscriptions: {total_subscriptions}")
print(f"Subscription Rate: {test_sub_rate:.2f}%")
print(f"Actions per Episode: {actions_per_episode:.1f}")
print(f"\nImprovement over random (0.44%): {test_sub_rate / 0.44:.1f}x")

# Compare with baseline
baseline_rate = 1.30  # From original Q-Learning baseline
tabular_fs_rate = 0.80  # From tabular Q-Learning feature selection

print(f"\nComparison:")
print(f"  Baseline (Q-Learning, 15 features): {baseline_rate:.2f}%")
print(f"  Feature Selection (Tabular): {tabular_fs_rate:.2f}% ❌ FAILED")
print(f"  Feature Selection (DQN): {test_sub_rate:.2f}% ✅")

if test_sub_rate > baseline_rate:
    print(f"\n🎉 DQN Feature Selection BEATS Baseline by {test_sub_rate - baseline_rate:.2f}%!")
elif test_sub_rate > tabular_fs_rate:
    print(f"\n✅ DQN Feature Selection FIXES the state space explosion (0.80% → {test_sub_rate:.2f}%)")
else:
    print(f"\n⚠️ DQN needs tuning (try more training steps or different hyperparameters)")

# Feature importance
print(f"\nMost Toggled Features:")
feature_names = [
    'Country', 'Stage', 'Status', 'Days_First', 'Days_Last', 'Days_Between',
    'Contact_Freq', 'Had_Call', 'Had_Demo', 'Had_Survey', 'Had_Signup',
    'Had_Manager', 'Country_ConvRate', 'Education_ConvRate', 'Stages_Completed'
]
top_features = np.argsort(feature_usage)[::-1][:5]
for i, feat_idx in enumerate(top_features, 1):
    print(f"  {i}. {feature_names[feat_idx]}: {int(feature_usage[feat_idx])} toggles")
```

---

## Summary

**Current Status:** ✅ Tabular Q-Learning working well for 15-dim baseline (1.30%, 3.0x improvement)

**Next Step:** 🚀 Implement DQN for 30-dim feature selection (expected 1.5-2.0%)

**Long-Term Path:**
1. DQN basics (1 hour)
2. Double DQN, Dueling DQN improvements (1-2 days)
3. PPO for continuous actions (3-5 days)
4. Advanced topics as needed (weeks/months)

**Why this order:**
- DQN solves your immediate problem (state space explosion)
- Shows progression: Tabular → Deep RL
- Demonstrates breadth of RL knowledge
- Strong interview talking points

**Key insight:** Don't chase fancy algorithms - use the simplest one that works. Your baseline (tabular Q-Learning) is perfect for 15-dim state. Only use DQN when state space is too large (30-dim, 45-dim). Only use PPO when you need continuous actions or multi-objective optimization.

---

## For Interviews

**Q: "What would you do next to improve your RL model?"**

**A:** "My immediate next step is implementing DQN for feature selection:

**Current bottleneck:**
- Feature selection failed with tabular Q-Learning (0.80% vs 1.30% baseline)
- Root cause: 30-dim state → 522K states → sparse Q-table
- Tabular Q-Learning can't generalize across similar states

**DQN solution:**
- Replace Q-table with neural network
- Network learns: 'State [0.5, 0.3, ...] is similar to [0.52, 0.29, ...]'
- Can handle 30-dim state → Expected 1.5-2.0% performance
- Implementation: 1 hour with Stable-Baselines3

**Long-term:**
- If I needed continuous actions ('wait X days' where X ∈ [0, 7]): Use PPO
- If I had multi-objective (subscriptions + satisfaction): Use PPO with custom reward
- If baseline was still best: Use simpler algorithm (occam's razor)

**Key lesson:** Use simplest algorithm that works. Tabular Q-Learning is perfect for 15-dim state (1.30%). Only use deep RL when state space exceeds ~10K states."
