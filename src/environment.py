"""
CRM Sales Funnel Gymnasium Environment

NUANCE #3: Class Imbalance Handling (228:1)
This environment implements batch-level oversampling to handle extreme imbalance:
- Natural: 0.44% subscription rate (agent almost never sees success)
- Solution: 30/30/40 batch sampling (agent sees success 30% of time)

NUANCE #4: Reward Shaping for Sparse Rewards
- Terminal reward: +100 (subscription)
- Intermediate rewards: +15 (call), +12 (demo), etc.
- All intermediate < 25% of terminal (prevents reward hacking)

INTERVIEW PREP: Be able to explain WHY batch-level vs dataset-level oversampling!
"""

import logging
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class CRMSalesFunnelEnv(gym.Env):
    """
    Gymnasium environment for CRM sales pipeline reinforcement learning.

    Implements modular "input -> decision box -> output" framework:
    - Input modules: Customer data, temporal features, pipeline position
    - Decision box: RL agent (Q-Learning)
    - Output: Action + consequences (rewards, state transitions)

    State Space: 16-dimensional continuous vector
    Action Space: 6 discrete actions (Email, Call, Demo, Survey, Wait, Manager)
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, data_path='data/processed/crm_train.csv',
                 stats_path='data/processed/historical_stats.json',
                 mode='train'):
        """
        Initialize environment.

        Args:
            data_path: Path to processed CSV (train/val/test)
            stats_path: Path to historical statistics JSON
            mode: 'train', 'val', or 'test' (affects batch sampling)
        """
        super().__init__()

        self.mode = mode

        # Load data
        logger.info(f"Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)

        # Load historical stats
        with open(stats_path, 'r') as f:
            self.historical_stats = json.load(f)

        # NUANCE #3: Separate customer pools for batch oversampling
        # WHY: Enables 30/30/40 sampling strategy
        self.subscribed_customers = self.df[self.df['Subscribed_Binary'] == 1].copy()
        self.first_call_customers = self.df[self.df['Had_First_Call'] == 1].copy()
        self.all_customers = self.df.copy()

        logger.info("="*80)
        logger.info(f"CRM SALES FUNNEL ENVIRONMENT INITIALIZED ({mode} mode)")
        logger.info("="*80)
        logger.info(f"Total customers: {len(self.df):,}")
        logger.debug(f"Subscribed: {len(self.subscribed_customers):,} " +
              f"({len(self.subscribed_customers)/len(self.df)*100:.2f}%)")
        logger.debug(f"Had first call: {len(self.first_call_customers):,} " +
              f"({len(self.first_call_customers)/len(self.df)*100:.2f}%)")
        logger.debug(f"Class imbalance: {(len(self.df)-len(self.subscribed_customers))/len(self.subscribed_customers):.0f}:1")
        logger.info("="*80)

        # ACTION SPACE: 6 discrete actions
        # NUANCE: Costs reflect real sales team effort/expense
        self.action_space = spaces.Discrete(6)
        self.action_names = {
            0: "Send Email",              # Low cost, low commitment
            1: "Make Phone Call",         # Medium cost, medium commitment
            2: "Schedule Demo",           # High cost, high commitment
            3: "Send Survey",             # Low cost, feedback gathering
            4: "No Action (Wait)",        # No cost, observe
            5: "Assign Account Manager"   # Highest cost, personalized service
        }
        self.action_costs = {
            0: -1,   # Email: $1
            1: -5,   # Call: $5
            2: -10,  # Demo: $10
            3: -2,   # Survey: $2
            4: 0,    # Wait: $0
            5: -20   # Manager: $20
        }

        # OBSERVATION SPACE: 15-dimensional continuous state
        # [0-103]: Country (104 categories)
        # [0-6]: Stage
        # [0-1]: Status, temporal features, binary flags, conv rates
        # NOTE: Education_Encoded removed (unordered bootcamp aliases, not ordinal)
        self.observation_space = spaces.Box(
            low=np.array([0]*15, dtype=np.float32),
            high=np.array([103, 6, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 5], dtype=np.float32),
            dtype=np.float32
        )

        # Episode configuration
        self.max_steps = 10  # Max actions per customer

        # Episode state (reset in reset())
        self.current_customer = None
        self.current_customer_state = None  # Store original state
        self.current_stage = 0
        self.steps_taken = 0
        self.actions_history = []

    def reset(self, seed=None, options=None):
        """
        Reset environment and return initial state.

        NUANCE #3: Batch-level oversampling
        - 30% subscribed customers (positive examples)
        - 30% first call customers (partial success)
        - 40% random (mostly negatives)

        WHY: Agent needs to see success examples to learn!
        Without this, agent sees success 0.44% of time (never learns).

        Returns:
            state: 16-dim numpy array
            info: Dict with customer metadata
        """
        super().reset(seed=seed)

        # BATCH SAMPLING STRATEGY (30/30/40)
        # INTERVIEW QUESTION: "Why batch-level instead of dataset-level oversampling?"
        # ANSWER: Dataset-level loses diversity, batch-level preserves it while
        # ensuring agent sees enough positive examples during training

        sample_type = np.random.rand()

        if self.mode == 'train':
            # Training: Use batch oversampling
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
            # Validation/Test: Natural distribution (no oversampling)
            self.current_customer = self.all_customers.sample(n=1).iloc[0]

        # Store original customer state for reference
        self.current_customer_state = self.current_customer.copy()

        # Initialize episode state
        self.current_stage = int(self.current_customer['Stage_Encoded'])
        self.steps_taken = 0
        self.actions_history = []

        # Get initial state
        state = self._get_state()

        info = {
            'customer_id': self.current_customer.get('ID', -1),
            'subscribed': int(self.current_customer['Subscribed_Binary']),
            'had_call': int(self.current_customer['Had_First_Call']),
            'initial_stage': self.current_stage
        }

        return state, info

    def step(self, action):
        """
        Execute action and return (state, reward, done, truncated, info).

        NUANCE #4: Reward Shaping
        - Immediate costs (action expenses)
        - Intermediate rewards (stage progressions)
        - Terminal reward (subscription achieved)

        Args:
            action: Integer 0-5

        Returns:
            next_state: 16-dim array
            reward: Float
            done: Bool (episode complete - subscription)
            truncated: Bool (max steps reached)
            info: Dict with step metadata
        """
        self.steps_taken += 1
        self.actions_history.append(action)

        # Initialize reward
        reward = 0
        previous_stage = self.current_stage

        # REWARD COMPONENT 1: Action costs (efficiency)
        # WHY: Encourages agent to minimize acquisition costs
        reward += self.action_costs[action]

        # REWARD COMPONENT 2: Intermediate rewards (progress)
        # NUANCE: These guide learning but don't dominate terminal reward
        # All intermediate < 25% of terminal (+100)

        stage_reward = 0

        # Action 1: Make Phone Call -> First Call stage
        # INTERVIEW: Why +15? Because first call has 11% conversion rate
        # (441 had calls, 48 subscribed -> 48/441 = 10.9%)
        if action == 1 and self.current_customer_state['Had_First_Call'] == 1:
            stage_reward = 15  # SECONDARY GOAL: First call achieved
            self.current_stage = max(self.current_stage, 1)

        # Action 2: Schedule Demo -> Demo stage
        elif action == 2 and self.current_customer_state['Had_Demo'] == 1:
            stage_reward = 12
            self.current_stage = max(self.current_stage, 2)

        # Action 3: Send Survey -> Survey stage
        elif action == 3 and self.current_customer_state['Had_Survey'] == 1:
            stage_reward = 8
            self.current_stage = max(self.current_stage, 3)

        # Action 5: Assign Manager -> Manager stage
        elif action == 5 and self.current_customer_state['Had_Manager'] == 1:
            stage_reward = 10
            self.current_stage = max(self.current_stage, 5)

        # Platform signup (high-value progression)
        if self.current_customer_state['Had_Signup'] == 1 and self.current_stage >= 4:
            stage_reward += 20

        reward += stage_reward

        # REWARD COMPONENT 3: Terminal reward (PRIMARY GOAL)
        # +100 for subscription (dominates all other rewards)
        # WHY: This is the actual business objective!

        done = False

        if self.current_customer_state['Subscribed_Binary'] == 1:
            # Customer can subscribe if we've engaged them enough
            if self.current_stage >= 4:
                reward += 100  # PRIMARY GOAL ACHIEVED!
                done = True

        # REWARD COMPONENT 4: Episode truncation
        truncated = False
        if self.steps_taken >= self.max_steps:
            truncated = True
            if not done:
                # Failed to convert within max steps
                reward -= 10  # Inefficiency penalty

        # REWARD COMPONENT 5: Behavioral penalties
        # Prevent spam (repeating same action)
        if len(self.actions_history) >= 2:
            if self.actions_history[-1] == self.actions_history[-2] and action != 4:
                reward -= 5  # Spam penalty (except for "wait")

        # Get next state
        next_state = self._get_state()

        info = {
            'stage': self.current_stage,
            'stage_changed': self.current_stage != previous_stage,
            'steps': self.steps_taken,
            'action_name': self.action_names[action],
            'action_cost': self.action_costs[action],
            'stage_reward': stage_reward,
            'subscribed': int(self.current_customer_state['Subscribed_Binary']),
            'total_reward': reward
        }

        return next_state, reward, done, truncated, info

    def _get_state(self):
        """
        Extract 16-dimensional state vector from current customer.

        NUANCE #2: Time Series -> RL State Conversion
        - Original: Sequential customer journey over time
        - Transformed: Fixed 16-dim snapshot
        - Temporal info preserved: days_since, binary flags, current stage

        INTERVIEW: "How did you convert temporal data to RL states?"
        ANSWER: "I created a snapshot state vector encoding:
          1. Current position (stage)
          2. Historical completions (binary flags)
          3. Temporal context (days_since normalized features)"

        Returns:
            state: 16-dim numpy array
        """
        c = self.current_customer_state

        state = np.array([
            # DEMOGRAPHICS (1 feature) - Static, known at lead creation
            # NOTE: Education_Encoded removed (B1-B30 are unordered bootcamp aliases)
            c['Country_Encoded'],             # 0: Categorical 0-103

            # PIPELINE POSITION (2 features) - Current observable status
            self.current_stage,               # 1: 0-6 encoding
            c['Status_Active'],               # 2: Binary

            # TEMPORAL (4 features) - All historical (past events)
            c['Days_Since_First_Norm'],       # 3: [0, 1]
            c['Days_Since_Last_Norm'],        # 4: [0, 1]
            c['Days_Between_Norm'],           # 5: [0, 1]
            c['Contact_Frequency'],           # 6: Engagement frequency

            # BINARY FLAGS (5 features) - Completed pipeline stages
            c['Had_First_Call'],              # 7: 0 or 1
            c['Had_Demo'],                    # 8: 0 or 1
            c['Had_Survey'],                  # 9: 0 or 1
            c['Had_Signup'],                  # 10: 0 or 1
            c['Had_Manager'],                 # 11: 0 or 1

            # AGGREGATED STATS (2 features) - From train set ONLY
            c['Country_ConvRate'],            # 12: [0, 1]
            c['Education_ConvRate'],          # 13: [0, 1] (per-bootcamp conversion)

            # DERIVED (1 feature) - Total stages completed
            c['Stages_Completed']             # 14: 0-5

        ], dtype=np.float32)

        return state

    def render(self, mode='human'):
        """
        Render environment state (optional).
        """
        if mode == 'human':
            logger.info(f"Step {self.steps_taken}/{self.max_steps}")
            logger.info(f"Stage: {self.current_stage}")
            logger.debug(f"Actions: {[self.action_names[a] for a in self.actions_history]}")
            logger.debug(f"Subscribed: {self.current_customer_state['Subscribed_Binary']}")


if __name__ == "__main__":
    """
    Test environment functionality.

    INTERVIEW PREP: Be able to explain each component independently!
    """
    logger.info("="*80)
    logger.info("TESTING CRM SALES FUNNEL ENVIRONMENT")
    logger.info("="*80)

    # Create environment
    env = CRMSalesFunnelEnv(
        data_path='data/processed/crm_train.csv',
        stats_path='data/processed/historical_stats.json',
        mode='train'
    )

    logger.info(f"Action Space: {env.action_space}")
    logger.info(f"Observation Space: {env.observation_space}")

    logger.info("="*80)
    logger.info("TEST 1: Batch Oversampling (30/30/40)")
    logger.info("="*80)

    # Run 1000 resets to verify sampling distribution
    subscribed_count = 0
    first_call_count = 0

    for i in range(1000):
        state, info = env.reset()
        subscribed_count += info['subscribed']
        first_call_count += info['had_call']

    logger.info(f"Out of 1000 resets:")
    logger.info(f"  Subscribed: {subscribed_count} ({subscribed_count/10:.1f}%)")
    logger.debug(f"  Expected: ~300 (30%) with batch oversampling")
    logger.debug(f"  Natural rate: ~4 (0.44%) without oversampling")
    logger.info(f"First Call: {first_call_count} ({first_call_count/10:.1f}%)")
    logger.debug(f"  Expected: ~500-600 (50-60%) with batch oversampling")

    logger.info("="*80)
    logger.info("TEST 2: Episode Execution")
    logger.info("="*80)

    # Run one episode
    state, info = env.reset()
    logger.debug(f"Initial state: {state}")
    logger.info(f"Customer subscribed: {info['subscribed']}")

    total_reward = 0
    done, truncated = False, False
    step = 0

    while not (done or truncated):
        # Random action
        action = env.action_space.sample()
        next_state, reward, done, truncated, step_info = env.step(action)

        total_reward += reward
        step += 1

        logger.debug(f"Step {step}: {step_info['action_name']}")
        logger.debug(f"  Reward: {reward:.1f} (Cost: {step_info['action_cost']}, " +
              f"Stage: {step_info['stage_reward']})")
        logger.debug(f"  Stage: {step_info['stage']}")

        state = next_state

    logger.info(f"Episode finished!")
    logger.info(f"  Total reward: {total_reward:.1f}")
    logger.info(f"  Steps: {step}")
    logger.debug(f"  Done: {done} (subscription achieved)")
    logger.debug(f"  Truncated: {truncated} (max steps)")

    logger.info("="*80)
    logger.info("TEST 3: Reward Range Verification")
    logger.info("="*80)

    # Check reward bounds
    min_reward = -10 * 20 - 10  # 10 steps * max cost + failure penalty
    max_reward = 100 + 20 + 15 + 12 + 10 + 8  # Terminal + all intermediate

    logger.debug(f"Theoretical reward range: [{min_reward}, {max_reward}]")
    logger.debug(f"Expected typical: [-100, +150]")

    logger.info("Environment tests complete!")
    logger.info("Ready for Q-Learning agent training!")
    logger.info("="*80)
