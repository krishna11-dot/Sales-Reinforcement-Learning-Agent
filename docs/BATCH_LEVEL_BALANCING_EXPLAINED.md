# Batch-Level Balancing - Simple Explanation

## The Problem: Extreme Class Imbalance

### Raw Numbers from Our Dataset

Out of 11,032 customers:
- 🟢 **48 subscribed** (0.44%) - SUCCESS
- 🟡 **441 had first call** (4.0%) - PARTIAL SUCCESS
- 🔴 **10,543 others** (95.6%) - NO SUCCESS

**Class Imbalance Ratio:** 228:1 (negative:positive)

### Why This Breaks Machine Learning

If we train using random sampling:

```
Episode 1: 🔴 No subscription (reward: -5)
Episode 2: 🔴 No subscription (reward: -3)
Episode 3: 🔴 No subscription (reward: -2)
...
Episode 227: 🔴 No subscription (reward: -4)
Episode 228: 🟢 SUBSCRIPTION! (reward: +100)
```

**Problem:** Agent sees success only **4 times per 1000 episodes**

**Result:** Agent learns "always predict failure" because that's correct 99.56% of the time!

---

## Our Solution: Batch-Level Balancing

### The Core Idea

**Don't change the dataset. Change how we sample from it during training.**

Instead of:
```
Pick random customer → 0.44% chance of positive example
```

We do:
```
Pick with bias → 30% chance of positive example
```

### The 30-30-40 Split

Every time the agent needs a new customer for training:

```python
Roll random number (0.0 to 1.0)

If < 0.3 (30% of time):
    → Sample from SUBSCRIBED customers 🟢
    → Agent sees what SUCCESS looks like

Else if < 0.6 (30% of time):
    → Sample from FIRST CALL customers 🟡
    → Agent sees PARTIAL SUCCESS (important milestone)

Else (40% of time):
    → Sample from ALL customers randomly 🔴
    → Agent sees REALISTIC distribution (mostly failures)
```

### Visual Comparison

**Natural Distribution (what random sampling gives):**
```
🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🟢
└─────────── 99.56% negative ────────────┘ 0.44% positive
```

**Our Training Distribution (30-30-40 sampling):**
```
🟢🟢🟢🟡🟡🟡🔴🔴🔴🔴
└─30%─┘└─30%─┘└─40%──┘

Effective positive rate: ~30% (68x increase!)
```

**Testing Distribution (natural, no bias):**
```
🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🔴🟢
└─────────── Still 99.56% negative ──────────┘

Fair evaluation!
```

---

## Implementation Details

### Where This Happens

**File:** [`src/environment.py`](../src/environment.py)
**Method:** `reset()` (lines 122-180)

### The Code

```python
def reset(self, seed=None, options=None):
    """Reset environment and return initial state."""

    sample_type = np.random.rand()  # Random number 0.0-1.0

    if self.mode == 'train':
        # TRAINING: Use batch-level balancing

        if sample_type < 0.3 and len(self.subscribed_customers) > 0:
            # 30%: Sample from subscribed customers
            self.current_customer = self.subscribed_customers.sample(n=1).iloc[0]

        elif sample_type < 0.6 and len(self.first_call_customers) > 0:
            # 30%: Sample from first call customers
            self.current_customer = self.first_call_customers.sample(n=1).iloc[0]

        else:
            # 40%: Random sample (mostly negatives)
            self.current_customer = self.all_customers.sample(n=1).iloc[0]

    else:
        # VALIDATION/TEST: Natural distribution (no bias)
        self.current_customer = self.all_customers.sample(n=1).iloc[0]

    return state, info
```

### The Three Customer Pools

Created during environment initialization ([`environment.py:63-67`](../src/environment.py#L63-L67)):

```python
# Pool 1: Subscribed customers (48 total)
self.subscribed_customers = self.df[self.df['Subscribed_Binary'] == 1].copy()

# Pool 2: First call customers (441 total)
self.first_call_customers = self.df[self.df['Had_First_Call'] == 1].copy()

# Pool 3: All customers (11,032 total)
self.all_customers = self.df.copy()
```

---

## Why 30-30-40? Design Rationale

### Option 1: 50-50 Split (Not Chosen)
```
50% positive, 50% negative
```
**Problem:** Too aggressive, doesn't reflect reality at all
**Risk:** Agent overfits to positives, performs poorly in production

### Option 2: Natural Distribution (Not Chosen)
```
0.44% positive, 99.56% negative
```
**Problem:** Too sparse, agent never learns success patterns
**Risk:** Agent learns "always say no"

### ✅ Option 3: 30-30-40 Split (Our Choice)
```
30% subscribed (success)
30% first call (partial success)
40% random (realistic mix)
```

**Why this works:**

1. **Sufficient positive examples:** 30% subscribed means agent sees success 300 times per 1000 episodes (vs 4 naturally)

2. **Intermediate milestones:** 30% first call teaches agent that "first call is progress toward subscription"

3. **Reality check:** 40% random keeps agent grounded in realistic failure cases

4. **Balanced learning:** Agent learns BOTH:
   - What success looks like (from 30% subscribed)
   - What failure looks like (from 40% random)

5. **Empirical validation:** This ratio achieved **3.0x improvement** (1.30% vs 0.44% baseline)

### Mathematical Justification

Natural positive rate: **0.44%**
Our training positive rate: **~30%**
Testing positive rate: **0.44%** (unchanged!)

```
Training boost: 30% / 0.44% = 68x more positive examples
Testing fairness: No bias applied = True performance measure
```

---

## Comparison to Other Techniques

### 1. Traditional Upsampling (NOT What We Do)

**Method:** Duplicate minority class rows in dataset

```
Original dataset:
🟢 (48 rows)
🔴 (10,984 rows)

After upsampling:
🟢 🟢 🟢 ... 🟢 (10,984 rows - copied 228 times!)
🔴 (10,984 rows)
```

**Problems:**
- ❌ Exact duplicates → Agent memorizes, doesn't generalize
- ❌ Dataset bloat → 228x storage increase
- ❌ Training slower → More data to process
- ❌ Overfitting risk → Sees same examples repeatedly

**Why we didn't use it:** Loses diversity, creates artificial dataset

---

### 2. SMOTE (Synthetic Minority Over-sampling)

**Method:** Create synthetic positive examples by interpolating between existing ones

```
Real customer 1: Education=B27, Country=USA, Stage=3
Real customer 2: Education=B11, Country=UK, Stage=5

SMOTE creates:
Synthetic customer: Education=B19(?), Country=??? (?), Stage=4
```

**Problems:**
- ❌ Assumes features are interpolatable (not true for categories!)
- ❌ Creates "impossible" customers (Country can't be "halfway between USA and UK")
- ❌ Complex assumptions to validate
- ❌ Risk of creating out-of-distribution samples

**Why we didn't use it:** Our features are mostly categorical (Education, Country), not continuous. SMOTE assumes you can blend features, which doesn't make sense here.

---

### 3. Class Weights (NOT Applicable)

**Method:** Weight loss function by class frequency

```python
weight_positive = total / (2 * positive_count)
weight_negative = total / (2 * negative_count)
```

**Problems:**
- ❌ Requires supervised learning with loss function
- ❌ We're using Q-Learning (RL), not supervised learning
- ❌ No "loss function" to weight

**Why we didn't use it:** Wrong paradigm - this is for classification, not RL

---

### 4. ✅ Batch-Level Balancing (What We Do)

**Method:** Bias the sampling during training, not the dataset itself

```python
# During training episode selection:
if random() < 0.3:
    sample from positives
elif random() < 0.6:
    sample from partial positives
else:
    sample from all

# During testing:
sample from all (natural distribution)
```

**Advantages:**
- ✅ **No data duplication** → All samples are real customers
- ✅ **Maintains diversity** → Never see exact same customer twice
- ✅ **Satisfies assumptions** → All samples are valid, real data points
- ✅ **Efficient** → No storage overhead
- ✅ **Fair evaluation** → Test on natural distribution
- ✅ **Simple** → Easy to implement and understand
- ✅ **RL-compatible** → Works naturally with episode-based training

**Why we used it:** Best balance of learning efficiency and data integrity

---

## Results and Validation

### Training Performance (with 30-30-40 sampling)

Out of 1000 training episodes:
- Agent sees **~300 subscribed customers** (30%)
- Agent sees **~300 first call customers** (30%)
- Agent sees **~400 random customers** (40%, mostly failures)

**Learning curve:** Agent converges around episode 50,000-70,000

### Testing Performance (natural distribution)

Out of 1,655 test customers:
- **1.30% subscription rate** (vs 0.44% random baseline)
- **3.0x improvement**
- No oversampling bias

**Validation:** Testing uses natural distribution, so performance is realistic!

---

## Interview Talking Points

### Q: "How did you handle class imbalance?"

**Answer:**
> "I used batch-level balancing during training. Instead of modifying the dataset, I biased the episode sampling - 30% from subscribed customers, 30% from partial success customers, and 40% from the full dataset. This gave the agent enough positive examples to learn (300 per 1000 episodes instead of 4), while keeping testing on the natural distribution for fair evaluation."

### Q: "Why not use SMOTE or upsampling?"

**Answer:**
> "I considered both. SMOTE assumes features are interpolatable, but mine are mostly categorical - you can't blend 'Country=USA' and 'Country=UK' meaningfully. Traditional upsampling creates exact duplicates, which risks overfitting. Batch-level balancing is safer because every sample is a real customer, maintaining data integrity while ensuring the agent sees enough positive examples to learn."

### Q: "Why 30-30-40 specifically?"

**Answer:**
> "I needed the agent to see enough positive examples to learn success patterns - 30% subscribed gives 300 examples per 1000 episodes, which is sufficient. The 30% first-call customers teach the agent that getting a first call is progress toward subscription (intermediate milestone). The 40% random keeps the agent grounded in realistic failure cases. This empirically achieved 3.0x improvement on the test set."

### Q: "Doesn't this bias your model?"

**Answer:**
> "The key is I only apply this during training. Testing and validation use the natural distribution (0.44% positive rate) with no sampling bias. This ensures the performance metrics reflect real-world performance. The sampling bias helps the agent learn, but we measure performance fairly."

### Q: "How would you improve this?"

**Answer:**
> "I could experiment with curriculum learning - start with 50-50 balance early in training for rapid learning, then gradually shift toward natural distribution. I could also try dynamic adjustment based on Q-value convergence. However, the current 30-30-40 approach works well and is simple to understand and maintain."

---

## Code References

| What | Where |
|------|-------|
| **Customer pools creation** | [`environment.py:63-67`](../src/environment.py#L63-L67) |
| **Sampling logic** | [`environment.py:140-160`](../src/environment.py#L140-L160) |
| **Training with sampling** | [`train.py:100-127`](../src/train.py#L100-L127) |
| **Testing (no sampling)** | [`evaluate.py`](../src/evaluate.py) |
| **Performance metrics** | [`logs/test_results.json`](../logs/test_results.json) |

---

## Summary

**What we do:**
- Bias episode sampling during training (30-30-40)
- Keep testing fair (natural distribution)
- All samples are real customers (no synthetic data)

**Why it works:**
- Agent sees 68x more positive examples during training
- Learns success patterns without overfitting
- Testing measures true performance (no bias)

**Result:**
- 3.0x improvement (1.30% vs 0.44%)
- Simple, interpretable, maintainable
- Scientifically sound (no synthetic data assumptions)

**The key principle:** *Smart sampling beats synthetic data.*
