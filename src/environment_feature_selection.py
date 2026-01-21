"""
CRM Sales Funnel Gymnasium Environment WITH FEATURE SELECTION

CRITICAL UPDATE: Implements Option 1 - Feature Mask Approach
- State: [feature_mask (15 binary), customer_features (15 continuous)] = 30 dimensions
- Actions: [Toggle_Feature_0...14, Email, Call, Demo, Survey, Wait, Manager] = 21 actions
- Agent learns BOTH which features to use AND which CRM actions to take
- NOTE: Education_Encoded removed (B1-B30 are unordered bootcamp aliases)

REQUIREMENT SATISFACTION:
"State space comprises all possible subsets of the features"
This implementation allows the agent to select which customer attributes matter.

FILE LOCATION: src/environment_feature_selection.py
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


class CRMFeatureSelectionEnv(gym.Env):
    """
    Gymnasium environment for CRM sales pipeline WITH FEATURE SELECTION.

    KEY DIFFERENCES from original environment.py:
    1. State space: 30 dimensions (15 mask + 15 features) instead of 15
    2. Action space: 21 actions (15 toggles + 6 CRM) instead of 6
    3. Episode flow: Agent can toggle features BEFORE taking CRM action
    4. Reward includes complexity penalty (prefer fewer features)

    DECISION BOX FLOW:
    Input -> [Feature Mask, Customer Data] -> Agent Decision ->
    [Toggle Feature OR CRM Action] -> Output [Reward, Next State]
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, data_path='data/processed/crm_train.csv',
                 stats_path='data/processed/historical_stats.json',
                 mode='train'):
        """
        Initialize environment with feature selection capability.

        Args:
            data_path: Path to processed CSV
            stats_path: Path to historical statistics JSON
            mode: 'train', 'val', or 'test'
        """
        super().__init__()

        self.mode = mode
        self.n_features = 15  # Number of customer features (Education_Encoded removed)
        self.n_crm_actions = 6  # Email, Call, Demo, Survey, Wait, Manager
        self.n_toggle_actions = 15  # One toggle per feature

        # Load data
        logger.info(f"Loading data from: {data_path}")
        self.df = pd.read_csv(data_path)

        # Load historical stats
        with open(stats_path, 'r') as f:
            self.historical_stats = json.load(f)

        # Separate customer pools for batch oversampling
        self.subscribed_customers = self.df[self.df['Subscribed_Binary'] == 1].copy()
        self.first_call_customers = self.df[self.df['Had_First_Call'] == 1].copy()
        self.all_customers = self.df.copy()

        logger.info("="*80)
        logger.info(f"CRM FEATURE SELECTION ENVIRONMENT INITIALIZED ({mode} mode)")
        logger.info("="*80)
        logger.info(f"Total customers: {len(self.df):,}")
        logger.debug(f"Subscribed: {len(self.subscribed_customers):,} " +
              f"({len(self.subscribed_customers)/len(self.df)*100:.2f}%)")
        logger.debug(f"Had first call: {len(self.first_call_customers):,} " +
              f"({len(self.first_call_customers)/len(self.df)*100:.2f}%)")
        logger.debug(f"Class imbalance: {(len(self.df)-len(self.subscribed_customers))/len(self.subscribed_customers):.0f}:1")
        logger.info("="*80)

        # UPDATED ACTION SPACE: 21 actions total
        # 0-14: Toggle features (feature selection)
        # 15-20: CRM actions (customer interaction)
        self.action_space = spaces.Discrete(21)

        self.action_names = {
            # Feature toggle actions (0-14) - Education removed
            0: "Toggle_Country",
            1: "Toggle_Stage",
            2: "Toggle_Status",
            3: "Toggle_Days_Since_First",
            4: "Toggle_Days_Since_Last",
            5: "Toggle_Days_Between",
            6: "Toggle_Contact_Freq",
            7: "Toggle_Had_First_Call",
            8: "Toggle_Had_Demo",
            9: "Toggle_Had_Survey",
            10: "Toggle_Had_Signup",
            11: "Toggle_Had_Manager",
            12: "Toggle_Country_ConvRate",
            13: "Toggle_Education_ConvRate",
            14: "Toggle_Stages_Completed",

            # CRM actions (15-20)
            15: "Send Email",
            16: "Make Phone Call",
            17: "Schedule Demo",
            18: "Send Survey",
            19: "No Action (Wait)",
            20: "Assign Account Manager"
        }

        # CRM action costs (same as before)
        self.action_costs = {
            15: -1,   # Email
            16: -5,   # Call
            17: -10,  # Demo
            18: -2,   # Survey
            19: 0,    # Wait
            20: -20   # Manager
        }

        # UPDATED OBSERVATION SPACE: 30 dimensions
        # First 15: Feature mask (binary 0/1) - Education_Encoded removed
        # Last 15: Customer features (continuous)
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(30,),  # 15 mask + 15 features
            dtype=np.float32
        )

        # Episode configuration
        self.max_steps = 20  # Allow more steps for feature exploration

        # Episode state
        self.current_customer = None
        self.current_customer_state = None
        self.current_stage = 0
        self.steps_taken = 0
        self.actions_history = []
        self.feature_mask = None  # NEW: Track which features are active
        self.feature_toggles_count = 0  # NEW: Count feature toggles

        # Feature names for analysis (Education removed - unordered bootcamp aliases)
        self.feature_names = [
            'Country', 'Stage', 'Status_Active',
            'Days_Since_First_Norm', 'Days_Since_Last_Norm',
            'Days_Between_Norm', 'Contact_Frequency',
            'Had_First_Call', 'Had_Demo', 'Had_Survey',
            'Had_Signup', 'Had_Manager', 'Country_ConvRate',
            'Education_ConvRate', 'Stages_Completed'
        ]

    def reset(self, seed=None, options=None):
        """
        Reset environment and return initial state.

        NEW BEHAVIOR:
        - Initialize feature mask (all features ON by default)
        - Return 30-dim state: [mask (15), features (15)]

        Returns:
            state: 30-dim numpy array
            info: Dict with customer metadata
        """
        super().reset(seed=seed)

        # Sample customer (batch oversampling for training)
        sample_type = np.random.rand()

        if self.mode == 'train':
            if sample_type < 0.3 and len(self.subscribed_customers) > 0:
                self.current_customer = self.subscribed_customers.sample(n=1).iloc[0]
            elif sample_type < 0.6 and len(self.first_call_customers) > 0:
                self.current_customer = self.first_call_customers.sample(n=1).iloc[0]
            else:
                self.current_customer = self.all_customers.sample(n=1).iloc[0]
        else:
            self.current_customer = self.all_customers.sample(n=1).iloc[0]

        # Store original customer state
        self.current_customer_state = self.current_customer.copy()

        # Initialize episode state
        self.current_stage = int(self.current_customer['Stage_Encoded'])
        self.steps_taken = 0
        self.actions_history = []
        self.feature_toggles_count = 0

        # NEW: Initialize feature mask
        # Strategy: Start with ALL features ON
        # Agent must learn which to turn OFF (pruning approach)
        self.feature_mask = np.ones(self.n_features, dtype=np.float32)

        # Alternative: Start with ALL features OFF
        # Agent must learn which to turn ON (building approach)
        # self.feature_mask = np.zeros(self.n_features, dtype=np.float32)

        # Get initial state (32-dim)
        state = self._get_state()

        info = {
            'customer_id': self.current_customer.get('ID', -1),
            'subscribed': int(self.current_customer['Subscribed_Binary']),
            'had_call': int(self.current_customer['Had_First_Call']),
            'initial_stage': self.current_stage,
            'n_features_active': int(np.sum(self.feature_mask))
        }

        return state, info

    def step(self, action):
        """
        Execute action and return (state, reward, done, truncated, info).

        NEW BEHAVIOR:
        - Actions 0-14: Toggle features (non-terminal, episode continues)
        - Actions 15-20: CRM actions (terminal, episode ends)

        Args:
            action: Integer 0-20

        Returns:
            next_state: 30-dim array
            reward: Float
            done: Bool
            truncated: Bool
            info: Dict
        """
        self.steps_taken += 1
        self.actions_history.append(action)

        # Route to appropriate handler
        if action < self.n_toggle_actions:
            # Feature toggle action (0-14)
            return self._handle_feature_toggle(action)
        else:
            # CRM action (15-20)
            return self._handle_crm_action(action - self.n_toggle_actions)

    def _handle_feature_toggle(self, feature_idx):
        """
        Toggle a single feature on/off.

        IMPORTANT: This is a NON-TERMINAL action
        Episode continues so agent can keep toggling

        Args:
            feature_idx: 0-14 (which feature to toggle)

        Returns:
            state, reward, done, truncated, info
        """
        # Toggle the feature (1 -> 0 or 0 -> 1)
        self.feature_mask[feature_idx] = 1 - self.feature_mask[feature_idx]
        self.feature_toggles_count += 1

        # Calculate complexity penalty
        # Prefer fewer features (Occam's Razor)
        n_active_features = np.sum(self.feature_mask)
        complexity_penalty = -0.01 * n_active_features

        reward = complexity_penalty

        # Episode does NOT end
        done = False
        truncated = False

        # Check if max steps reached
        if self.steps_taken >= self.max_steps:
            truncated = True
            # Force agent to take CRM action by ending episode
            reward -= 20  # Penalty for not taking decisive action

        # Get new state
        next_state = self._get_state()

        info = {
            'action_type': 'feature_toggle',
            'feature_idx': feature_idx,
            'feature_name': self.feature_names[feature_idx],
            'feature_state': int(self.feature_mask[feature_idx]),
            'n_active_features': int(n_active_features),
            'stage': self.current_stage,
            'steps': self.steps_taken,
            'total_reward': reward
        }

        return next_state, reward, done, truncated, info

    def _handle_crm_action(self, crm_action):
        """
        Take a CRM action (Email, Call, Demo, Survey, Wait, Manager).

        IMPORTANT: This is a TERMINAL action
        Episode ends after this action

        Args:
            crm_action: 0-5 (mapped from actions 15-20)

        Returns:
            state, reward, done, truncated, info
        """
        # Map to original action indices (now 15-20 instead of 16-21)
        action_map = {0: 15, 1: 16, 2: 17, 3: 18, 4: 19, 5: 20}
        global_action = action_map[crm_action]

        previous_stage = self.current_stage
        reward = 0

        # REWARD COMPONENT 1: Action cost
        reward += self.action_costs[global_action]

        # REWARD COMPONENT 2: Simplicity bonus
        # Reward using fewer features
        n_active_features = np.sum(self.feature_mask)
        simplicity_bonus = -0.1 * n_active_features
        reward += simplicity_bonus

        # REWARD COMPONENT 3: Intermediate stage rewards
        # (Same logic as original environment)
        stage_reward = 0

        # Apply ONLY if corresponding feature is ACTIVE in mask
        # This forces agent to learn which features matter!

        if crm_action == 1 and self.current_customer_state['Had_First_Call'] == 1:
            # Only reward if Had_First_Call feature is active (index 7 after removing Education)
            if self.feature_mask[7] == 1:
                stage_reward = 15
                self.current_stage = max(self.current_stage, 1)

        elif crm_action == 2 and self.current_customer_state['Had_Demo'] == 1:
            if self.feature_mask[8] == 1:
                stage_reward = 12
                self.current_stage = max(self.current_stage, 2)

        elif crm_action == 3 and self.current_customer_state['Had_Survey'] == 1:
            if self.feature_mask[9] == 1:
                stage_reward = 8
                self.current_stage = max(self.current_stage, 3)

        elif crm_action == 5 and self.current_customer_state['Had_Manager'] == 1:
            if self.feature_mask[11] == 1:
                stage_reward = 10
                self.current_stage = max(self.current_stage, 5)

        if self.current_customer_state['Had_Signup'] == 1 and self.current_stage >= 4:
            if self.feature_mask[10] == 1:
                stage_reward += 20

        reward += stage_reward

        # REWARD COMPONENT 4: Terminal reward (subscription)
        done = False

        if self.current_customer_state['Subscribed_Binary'] == 1:
            if self.current_stage >= 4:
                reward += 100  # PRIMARY GOAL
                done = True

        # REWARD COMPONENT 5: Anti-spam penalty
        if len(self.actions_history) >= 2:
            if self.actions_history[-1] == self.actions_history[-2] and crm_action != 4:
                reward -= 5

        # Episode always ends after CRM action
        truncated = False

        # Get final state
        next_state = self._get_state()

        info = {
            'action_type': 'crm_action',
            'crm_action': crm_action,
            'action_name': self.action_names[global_action],
            'action_cost': self.action_costs[global_action],
            'stage_reward': stage_reward,
            'simplicity_bonus': simplicity_bonus,
            'n_active_features': int(n_active_features),
            'feature_toggles': self.feature_toggles_count,
            'stage': self.current_stage,
            'stage_changed': self.current_stage != previous_stage,
            'steps': self.steps_taken,
            'subscribed': int(self.current_customer_state['Subscribed_Binary']),
            'total_reward': reward,
            'active_features': [self.feature_names[i] for i in range(15) if self.feature_mask[i] == 1]
        }

        return next_state, reward, done, truncated, info

    def _get_state(self):
        """
        Extract 30-dimensional state vector.

        NEW FORMAT:
        state = [
            feature_mask[0...14],      # 15 binary values (which features active)
            customer_features[0...14]  # 15 continuous values (actual features)
        ]
        Note: Education_Encoded removed (B1-B30 are unordered bootcamp aliases)

        Returns:
            state: 30-dim numpy array
        """
        c = self.current_customer_state

        # Extract customer features (Education_Encoded removed - unordered bootcamp aliases)
        customer_features = np.array([
            c['Country_Encoded'],
            self.current_stage,
            c['Status_Active'],
            c['Days_Since_First_Norm'],
            c['Days_Since_Last_Norm'],
            c['Days_Between_Norm'],
            c['Contact_Frequency'],
            c['Had_First_Call'],
            c['Had_Demo'],
            c['Had_Survey'],
            c['Had_Signup'],
            c['Had_Manager'],
            c['Country_ConvRate'],
            c['Education_ConvRate'],
            c['Stages_Completed']
        ], dtype=np.float32)

        # Normalize customer features to [0, 1] for consistency
        # Country: 0-103 -> [0, 1]
        customer_features[0] /= 103.0
        # Stage: 0-6 -> [0, 1]
        customer_features[1] /= 6.0
        # Stages_Completed: 0-5 -> [0, 1]
        customer_features[14] /= 5.0

        # Concatenate: [mask (15), features (15)]
        state = np.concatenate([self.feature_mask, customer_features])

        return state

    def render(self, mode='human'):
        """
        Render environment state.
        """
        if mode == 'human':
            logger.info(f"Step {self.steps_taken}/{self.max_steps}")
            logger.info(f"Stage: {self.current_stage}")
            logger.info(f"Active features: {np.sum(self.feature_mask)}/15")
            active_feature_names = [self.feature_names[i] for i in range(15) if self.feature_mask[i] == 1]
            logger.debug(f"Features ON: {', '.join(active_feature_names[:5])}...")
            logger.debug(f"Actions: {[self.action_names[a] for a in self.actions_history[-3:]]}")


if __name__ == "__main__":
    """
    Test feature selection environment.
    """
    logger.info("="*80)
    logger.info("TESTING FEATURE SELECTION ENVIRONMENT")
    logger.info("="*80)

    # Create environment
    env = CRMFeatureSelectionEnv(
        data_path='data/processed/crm_train.csv',
        stats_path='data/processed/historical_stats.json',
        mode='train'
    )

    logger.info(f"Action Space: {env.action_space}")
    logger.info(f"Observation Space: {env.observation_space}")
    logger.info(f"Total Actions: 21 (15 feature toggles + 6 CRM actions)")

    logger.info("="*80)
    logger.info("TEST: Episode with Feature Selection")
    logger.info("="*80)

    state, info = env.reset()
    logger.debug(f"Initial state shape: {state.shape}")
    logger.debug(f"Feature mask (first 15): {state[:15]}")
    logger.debug(f"Customer features (last 15): {state[15:]}")
    logger.info(f"Active features: {info['n_features_active']}/15")

    # Simulate agent toggling features then taking action
    logger.info("--- Agent explores feature space ---")

    # Toggle Country OFF (action 0)
    state, reward, done, truncated, info = env.step(0)
    logger.debug(f"Step 1: {info['action_type']} - {info['feature_name']}")
    logger.debug(f"  Active features: {info['n_active_features']}/15")
    logger.debug(f"  Reward: {reward:.2f}")

    # Toggle Stage OFF (action 1)
    state, reward, done, truncated, info = env.step(1)
    logger.debug(f"Step 2: {info['action_type']} - {info['feature_name']}")
    logger.debug(f"  Active features: {info['n_active_features']}/15")
    logger.debug(f"  Reward: {reward:.2f}")

    # Take CRM action: Call (action 16 - now shifted)
    state, reward, done, truncated, info = env.step(16)
    logger.info(f"Step 3: {info['action_type']} - {info['action_name']}")
    logger.info(f"  Final active features: {info['n_active_features']}/15")
    logger.debug(f"  Features used: {', '.join(info['active_features'][:5])}...")
    logger.info(f"  Reward: {reward:.2f}")
    logger.info(f"  Done: {done} (subscribed: {info['subscribed']})")

    logger.info("="*80)
    logger.info("Feature selection environment ready!")
    logger.info("="*80)
