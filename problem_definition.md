# CRM Sales Pipeline RL Optimization - Problem Definition

## Business Context

**Company**: Education technology platform serving global customers
**Goal**: Optimize user acquisition pipeline to increase subscriptions from 0.44% to 1%+
**Data**: Internal CRM data (CONFIDENTIAL - do not share publicly)

**Dataset**: SalesCRM.xlsx
- 11,032 customers
- 13 columns (demographics, contact history, pipeline stages)
- 0.44% subscription rate (48 customers) - EXTREME 228:1 class imbalance
- 4.0% first call rate (441 customers)

## Project Questions

1. **WHO should sales team contact?** -> Identify high-value customer segments (Education x Country x Stage)
2. **WHAT actions lead to subscriptions?** -> Determine optimal action sequences

## Success Criteria

### Technical Metrics (for debugging/optimization)
- Q-value convergence: Delta Q < 0.001 over 10k episodes
- Action stability: < 5% policy change in final 10k episodes
- Epsilon decay: Smooth 1.0 -> 0.01 trajectory
- Episode rewards: Upward trend -> stabilization
- Q-table size: 5,000-10,000 visited states

### Business Metrics (for stakeholder value)
- **Subscription rate: 0.44% -> 1.0%+** (2.3x improvement)
- **First call rate: 4.0% -> 8.0%+** (2x improvement)
- **Cost per acquisition: 40% reduction** vs random baseline
- **ROI: Positive** (revenue > costs)
- **Customer segments: Top 3** high-value segments identified

**CRITICAL**: Both technical AND business metrics must succeed. Technical alone is insufficient.

## Key Design Decisions

### 1. Temporal Data Handling (No Leakage)
**Problem**: CRM data is time-ordered customer journeys
**Solution**:
- Split by DATE first (temporal split)
- Calculate statistics ONLY on train set
- Map train statistics to val/test sets

**Why**: Prevents test set outcomes from leaking into training features

### 2. Class Imbalance (228:1)
**Problem**: Natural sampling = 0.44% success rate, agent never learns
**Solution**: Batch-level oversampling (30/30/40)
- 30% subscribed customers
- 30% first call customers
- 40% random (mostly negatives)

**Why**: Agent sees success 30% of time, learns patterns, still exposed to failures

### 3. Reward Shaping
**Problem**: Pure sparse reward (only +100 at end) -> agent sees success 0.44% of episodes
**Solution**: Intermediate rewards
- First Call: +15 (11% conversion rate from this stage)
- Demo: +12
- Survey: +8
- Signup: +20
- Manager: +10
- Subscription: +100 (terminal)
- All intermediate < 25% of terminal

**Why**: Guides learning without dominating true objective

### 4. State Discretization
**Trade-off**: Precision vs learning speed
**Choice**: 2 decimals
- Days: 0.35 vs 0.36 ≈ 1-2 day difference (meaningful)
- Creates ~100 buckets per continuous feature
- Expected state space: 5,000-10,000 visited states

**Alternatives**:
- 3+ decimals: Sparse Q-table, slow learning
- 0-1 decimals: Lose information, poor decisions

### 5. Hyperparameters

**Learning Rate (alpha): 0.1**
- Conservative updates (10% step toward new estimate)
- Good for noisy environments (228:1 imbalance)
- Alternative alpha=0.5: Too fast, unstable

**Discount Factor (gamma): 0.95**
- Values future rewards highly (95% of immediate)
- Good for multi-step tasks (subscription takes 3-5 actions)
- Alternative gamma=0.5: Shortsighted, ignores long-term

**Epsilon Decay: 0.995**
- Reaches epsilon≈0.01 around episode 1000
- Fast initial exploration, gradual refinement
- Calculation: 0.01 = 1.0 × (0.995)^t -> t ≈ 919

## Critical Nuances for Interview

### Nuance #1: Project Goal Interpretation
**NOT**: Meta-learning to select which features to use in a model
**YES**: Learning which customer attributes predict subscription success

State = customer's features (WHO they are)
Actions = sales actions (WHAT to do)
Agent learns: "For customers with features X, action Y works best"

### Nuance #2: Time Series -> RL State Conversion
**Original**: Sequential events with timestamps
**Transformation**: Fixed 16-dim state vector
**Temporal handling**:
- Days_since_contact (normalized) replaces raw timestamps
- Binary flags capture cumulative history
- Current stage = position in funnel

### Nuance #3: Batch-Level vs Dataset-Level Oversampling
**Why batch-level?**
- Preserves data diversity
- Prevents overfitting
- Agent sees success examples during training
- Still exposed to failures

**Why NOT dataset-level?**
- Loses diversity
- Overfits to small positive class
- Duplicates same examples

### Nuance #4: Reward Hacking Prevention
All intermediate rewards < 25% of terminal reward

**Risk**: Agent might chase easy intermediate rewards (e.g., spam emails to get +15) instead of actual subscription

**Solution**: Design intermediate rewards proportional to conversion rates

### Nuance #5: Stage Column (NOT Data Leakage)
**Question**: Stage shows "subscribed already" - isn't that leakage?

**Answer**: NO - Stage is CURRENT STATUS, not future prediction
- Analogous to: "Patient currently in ICU" for predicting readmission
- Current status is observable NOW
- Not predicting the future

### Nuance #6: Separated Metrics
**Question**: Q-values converged. Is model good?

**Answer**: NO! Technical convergence != Business success

**Must report BOTH**:
- Technical: For debugging (Q-convergence, epsilon, etc.)
- Business: For stakeholders (conversion rate, ROI, etc.)

Both must succeed for project success.

### Nuance #7: Modular Design for Debugging
Each component testable independently:
- Environment: Do random actions work? Rewards in range?
- Agent: Are Q-values updating?
- Sampling: Is 30/30/40 split happening?
- Reward: Are intermediate rewards firing?

**If things break, can isolate which component failed**

## Interview Questions & Answers

### Q: "Walk me through your project"
2-min overview: Purpose (optimize sales), approach (Q-learning + batch oversampling), results (2.3x subscriptions), innovation (temporal handling + reward shaping)

### Q: "Why RL instead of supervised learning?"
RL optimizes sequential decisions with delayed rewards. Supervised learning predicts outcomes but doesn't tell you WHAT ACTION to take.

### Q: "What was the hardest part?"
Handling 228:1 class imbalance while avoiding leakage. Solution: Batch-level oversampling + temporal split.

### Q: "How would you improve this?"
- Deep RL (DQN) for continuous states
- Multi-agent (different agents for different customer segments)
- Online learning (adapt to changing customer behavior)

### Q: "Why not just use the most successful customer segment?"
That's supervised learning (prediction). We need RL to learn WHICH ACTIONS to take for each segment.

### Q: "How do you ensure no data leakage?"
1. Temporal split by date (train dates < test dates)
2. Statistics from train set only
3. Features use only historical data (past events)

### Q: "Why 30/30/40 split?"
Balances learning from successes (60%) with exposure to failures (40%). Pure oversampling (100% positive) would overfit.

### Q: "How do you know the agent learned something?"
Compare final policy to random baseline on test set. Measure business metrics (subscription rate, ROI) not just technical metrics.

## File Structure

### Data Files
- `data/raw/SalesCRM.xlsx` - Original data
- `data/processed/crm_train.csv` - Training set (70%)
- `data/processed/crm_val.csv` - Validation set (15%)
- `data/processed/crm_test.csv` - Test set (15%)
- `data/processed/historical_stats.json` - Statistics from train set only

### Source Files
- `src/data_processing.py` - Temporal-aware preprocessing
- `src/environment.py` - Gymnasium environment with batch oversampling
- `src/agent.py` - Q-Learning agent
- `src/train.py` - Training loop with separated metrics
- `src/evaluate.py` - Evaluation on test set

### Output Files
- `checkpoints/agent_final.pkl` - Trained model
- `logs/training_metrics_final.json` - All metrics
- `visualizations/training_curves.png` - Training plots

## State Space (16 Dimensions)

1-2. **Demographics**: Education, Country (categorical)
3-4. **Pipeline Position**: Stage, Status (ordinal, binary)
5-8. **Temporal**: Days since first/last contact, days between, frequency
9-13. **Binary Flags**: Had call, demo, survey, signup, manager
14-15. **Aggregated Stats**: Country conversion rate, Education conversion rate (from train set only)
16. **Derived**: Stages completed

## Action Space (6 Actions)

0. Send Email (cost: -$1)
1. Make Phone Call (cost: -$5)
2. Schedule Demo (cost: -$10)
3. Send Survey (cost: -$2)
4. No Action (cost: $0)
5. Assign Manager (cost: -$20)

## Implementation Details

- **Environment**: Gymnasium
- **Algorithm**: Tabular Q-Learning
- **Episodes**: 100,000
- **Expected runtime**: 30-90 minutes
- **Logging**: Every 1,000 episodes
- **Checkpoints**: Every 10,000 episodes

## Expected Results

- Subscription rate improvement: 2-3x baseline
- Cost per acquisition reduction: 30-50%
- Identified high-value segments
- Learned action sequences for conversion

## Next Steps After Implementation

1. Analyze learned policy (which actions for which customers)
2. Identify top customer segments (Education x Country x Stage)
3. Generate business recommendations
4. Create deployment plan
5. Design A/B test for production validation
